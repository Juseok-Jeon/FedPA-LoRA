import torch
import logging
import gc
try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
except:
    deepspeed = None
    DeepSpeedEngine = None
from federatedscope.register import register_trainer
from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.monitors.monitor import Monitor
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler
from federatedscope.llm.model.adapter_builder import AdapterModel
from federatedscope.llm.dataset.llm_dataset import DefaultToken   # added by me, for gsm8k evaluation
from federatedscope.llm.misc.fschat import FSChatBot_My   # added by me, for gsm8k evaluation
from federatedscope.llm.eval.eval_for_gsm8k.eval import *   # added by me, for gsm8k evaluation
from federatedscope.llm.dataset.llm_dataset import PROMPT_DICT   # added by me, for gsm8k evaluation

logger = logging.getLogger(__name__)


class LLMTrainer(GeneralTorchTrainer):
    def _hook_on_fit_start_numerical_precision(self, ctx):
        if self.cfg.train.is_enable_half:
            if not ctx.cfg.llm.deepspeed.use:
                ctx.model = ctx.model.half()

    def _hook_on_fit_start_init(self, ctx):
        if ctx.cfg.llm.deepspeed.use:
            # Enable deepspeed
            # TODO: save ctx.optimizer and ctx.scheduler
            # TODO: should clients share the same `ctx.model_engine`?
            assert deepspeed is not None, "Please install deepspeed."
            if not hasattr(ctx, 'model_engine'):
                ctx.model_engine, ctx.optimizer, _, ctx.scheduler = \
                    deepspeed.initialize(
                        config=ctx.cfg.llm.deepspeed.ds_config,
                        model=ctx.model,
                        model_parameters=filter(lambda p: p.requires_grad,
                                                ctx.model.parameters()),
                    )
            # Enable all cards from 0
            ctx.device = ctx.model_engine.local_rank
            if ctx.cfg.train.is_enable_half:
                ctx.fp16 = ctx.model_engine.fp16_enabled()
        else:
            # prepare model and optimizer
            ctx.model.to(ctx.device)
            if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
                # Initialize optimizer here to avoid the reuse of optimizers
                # across different routines
                ctx.optimizer = get_optimizer(
                    ctx.model, **ctx.cfg[ctx.cur_mode].optimizer)
                ctx.scheduler = get_scheduler(
                    ctx.optimizer, **ctx.cfg[ctx.cur_mode].scheduler)

        # prepare statistics
        ctx.loss_batch_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.loss_regular_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar([], LIFECYCLE.ROUTINE)
        
        # added by me, for gsm8k evaluation
        if not hasattr(ctx, 'val_loader_copy'):
            ctx.val_loader_copy = ctx.val_loader
        
    def _hook_on_batch_forward(self, ctx):
        if ctx.cur_mode == MODE.TEST and str(ctx.cfg.data.type).startswith('gsm8k'):
        # if ctx.cur_mode == MODE.TEST:
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            ctx.loss_regular = CtxVar(0., LIFECYCLE.BATCH)
            ctx.loss_batch = CtxVar(torch.tensor(0.0, device=ctx.device), LIFECYCLE.BATCH)
            ctx.batch_size = CtxVar(0, LIFECYCLE.BATCH)
            ctx.y_true = CtxVar(torch.tensor([], device=ctx.device), LIFECYCLE.BATCH)
            ctx.y_prob = CtxVar(torch.tensor([], device=ctx.device), LIFECYCLE.BATCH)
            return

        input_ids = ctx.data_batch['input_ids'].to(ctx.device)
        labels = ctx.data_batch['labels'].to(ctx.device)
        attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)

        if ctx.cur_mode == MODE.TEST:
            ctx.model.eval()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            with torch.inference_mode():
                if ctx.cfg.llm.deepspeed.use:
                    outputs = ctx.model_engine(input_ids=input_ids,
                                            labels=labels,
                                            attention_mask=attention_mask)
                else:
                    outputs = ctx.model(input_ids=input_ids,
                                        labels=labels,
                                        attention_mask=attention_mask)
                loss = outputs.loss.detach().cpu()
            ctx.loss_regular = CtxVar(0., LIFECYCLE.BATCH)
            ctx.skip_this_batch = CtxVar(torch.isnan(loss).item(), LIFECYCLE.BATCH)
            ctx.y_true = CtxVar(torch.tensor([]), LIFECYCLE.BATCH)
            ctx.y_prob = CtxVar(torch.tensor([]), LIFECYCLE.BATCH)
            ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
            ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)
            del outputs, input_ids, labels, attention_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return

        if ctx.cfg.llm.deepspeed.use:
            outputs = ctx.model_engine(input_ids=input_ids,
                                       labels=labels,
                                       attention_mask=attention_mask,
                                       use_cache=False)
        else:
            outputs = ctx.model(input_ids=input_ids,
                                labels=labels,
                                attention_mask=attention_mask,
                                use_cache=False)

        logits = outputs.logits
        loss = outputs.loss

        # Sec. 4.2 Global Product-Guided Alignment
        if getattr(ctx.cfg.lora, "regularization", False) and \
                getattr(ctx, "global_lora_ref", None) is not None:
            reg = 0.0
            ref = ctx.global_lora_ref
            for name, module in ctx.model.named_modules():
                if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                    for adapter in module.lora_A.keys():
                        B_key = f"{name}.lora_B.{adapter}.weight"
                        A_key = f"{name}.lora_A.{adapter}.weight"
                        B_ref_key = next(
                            (k for k in ref.keys()
                             if k.endswith(B_key) or B_key.endswith(k)),
                            None
                        )
                        A_ref_key = next(
                            (k for k in ref.keys()
                             if k.endswith(A_key) or A_key.endswith(k)),
                            None
                        )
                        
                        if B_ref_key is not None and A_ref_key is not None:
                            A = module.lora_A[adapter].weight.float()
                            B = module.lora_B[adapter].weight.float()
                            Ag = ref[A_ref_key].to(A.device).detach().float()
                            Bg = ref[B_ref_key].to(B.device).detach().float()
                            
                            E = B @ A - Bg @ Ag
                            reg = reg + E.pow(2).sum()

            loss_regular = 0.5 * float(ctx.cfg.lora.lam) * reg
            loss = loss + loss_regular
            ctx.loss_regular = CtxVar(loss_regular.detach().item(),
                                      LIFECYCLE.BATCH)
        else:
            ctx.loss_regular = CtxVar(0., LIFECYCLE.BATCH)

        if not torch.isfinite(loss):
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            ctx.loss_batch = CtxVar(torch.tensor(0.0, device=ctx.device),
                                    LIFECYCLE.BATCH)
            ctx.batch_size = CtxVar(0, LIFECYCLE.BATCH)
            ctx.y_true = CtxVar(torch.empty(0, device=ctx.device),
                                LIFECYCLE.BATCH)
            ctx.y_prob = CtxVar(torch.empty(0, device=ctx.device),
                                LIFECYCLE.BATCH)
            logger.warning('Skip NaN/Inf batch.')
            del outputs, logits, loss
            torch.cuda.empty_cache()
            return
        ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(logits, LIFECYCLE.BATCH)

        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

    def _hook_on_batch_backward(self, ctx):
        if ctx.skip_this_batch:
            return

        if ctx.cfg.llm.deepspeed.use:
            ctx.model_engine.backward(ctx.loss_task)
            ctx.model_engine.step()
        else:
            ctx.optimizer.zero_grad()
            ctx.loss_task.backward()

            if ctx.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                               ctx.grad_clip)

            ctx.optimizer.step()
        if ctx.scheduler is not None:
            ctx.scheduler.step()

    def _hook_on_batch_end(self, ctx):
        if ctx.skip_this_batch:
            if ctx.cfg.llm.retry_on_nan_loss:
                # Retry with new data in train and finetune
                if ctx.cur_mode == MODE.TRAIN:
                    self._run_batch(self.hooks_in_train, run_step=1)
                elif ctx.cur_mode == MODE.FINETUNE:
                    self._run_batch(self.hooks_in_ft, run_step=1)
            return

        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))

    def _hook_on_fit_end(self, ctx):
        avg_loss = 0 if float(
            ctx.num_samples) == 0 else ctx.loss_batch_total / float(
                ctx.num_samples)
        eval_results = {
                f'{ctx.cur_split}_loss': ctx.loss_batch_total,
                f'{ctx.cur_split}_total': ctx.num_samples,
                f'{ctx.cur_split}_avg_loss': avg_loss,
        }
        
        # added by me, evaluating on GSM8K dataset
        if ctx.cur_mode == MODE.TEST and str(ctx.cfg.data.type).startswith('gsm8k'):

            ctx.model.eval()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            fschatbot = FSChatBot_My(ctx.model, ctx.cfg)
            answers = []
            
            with torch.inference_mode():
                for batch in ctx.val_loader_copy:
                    for instruction, _, output in zip(batch['instruction'], batch['input'], batch['output']):
                        input_text = build_prompt(instruction, N_SHOT, COT_FLAG)
                        generate_kwargs = dict(max_new_tokens=256, top_p=0.95, temperature=0.8)
                        model_completion = fschatbot.generate(input_text, generate_kwargs)
                        model_answer = clean_answer(model_completion)
                        is_cor = is_correct(model_answer, output)
                        answers.append(is_cor)
                        print(f'Question: {instruction}\n\n'
                            f'Answers: {extract_answer_from_output(output)}\n\n'
                            f'Model Answers: {model_answer}\n\n'
                            f'Model Completion: {model_completion}\n\n'
                            f'Is correct: {is_cor}\n\n')

                        print(f'Num of total question: {len(answers)}, '
                            f'correct num: {sum(answers)}, '
                            f'correct rate: {float(sum(answers))/len(answers)}.')
                        
                        del model_completion, model_answer
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
            del fschatbot
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            eval_results[f'{ctx.cur_split}_acc'] = float(sum(answers))/len(answers)
        
        setattr(ctx, 'eval_metrics', eval_results)
                
        # # TODO: make this as a hook function
        # # Move trainable part to `cpu`, which can save memory but cost time
        # if ctx.cfg.llm.adapter.mv_to_cpu:
        #     for p in ctx.model.parameters():
        #         if p.requires_grad:
        #             p.data = p.to('cpu')
        #             if p.grad is not None:
        #                 p.grad.data = p.grad.to('cpu')

    def _hook_on_batch_forward_flop_count(self, ctx):
        """
        The monitoring hook to calculate the flops during the fl course

        Note:
          For customized cases that the forward process is not only \
          based on ctx.model, please override this function (inheritance \
          case) or replace this hook (plug-in case)

          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.monitor``                     Track average flops
            ==================================  ===========================
        """

        # The process may occupy a large amount of video memory
        # if the garbage collection is not triggered in time
        # when there is plenty of video memory left. Set
        # `eval.count_flops = False` to avoid this.
        if not isinstance(ctx.monitor, Monitor):
            logger.warning(
                f"The trainer {type(self)} does contain a valid monitor, "
                f"this may be caused by initializing trainer subclasses "
                f"without passing a valid monitor instance."
                f"Please check whether this is you want.")
            return

        if self.cfg.eval.count_flops and ctx.monitor.flops_per_sample == 0:
            # calculate the flops_per_sample
            try:
                input_ids = ctx.data_batch['input_ids'].to(ctx.device)
                labels = ctx.data_batch['labels'].to(ctx.device)
                attention_mask = ctx.data_batch['attention_mask'].to(
                    ctx.device)
                from fvcore.nn import FlopCountAnalysis
                if isinstance(ctx.model, AdapterModel):
                    flops_one_batch = FlopCountAnalysis(
                        ctx.model.model,
                        inputs=(input_ids, attention_mask)).total()
                else:
                    flops_one_batch = FlopCountAnalysis(
                        ctx.model, inputs=(input_ids, attention_mask)).total()
                ctx.monitor.track_avg_flops(flops_one_batch, ctx.batch_size)
            except Exception as e:
                logger.warning("When using count flops functions, torch's "
                               "garbage collection mechanism may not be "
                               "timely resulting in OOM, please set "
                               "`cfg.eval.count_flops` to `False` "
                               "to avoid error or warning like this.")
                logger.error(e)
                # Raise warning at the first failure
                logger.warning(
                    "current flop count implementation is for general LLM "
                    "trainer case: "
                    "1) ctx.data_batch contains [input_ids, labels, "
                    "attn_mask]; and 2) the ctx.model takes first two "
                    "arguments should be and attention_mask. "
                    "If ctx.model is an adapter model, the model in 2) has "
                    "been replaced by ctx.model.model. "
                    "Please check the forward format or implement your own "
                    "flop_count function")
                ctx.monitor.flops_per_sample = -1

        # by default, we assume the data has the same input shape,
        # thus simply multiply the flops to avoid redundant forward
        ctx.monitor.total_flops += ctx.monitor.flops_per_sample * \
            ctx.batch_size


def call_llm_trainer(trainer_type):
    if trainer_type == 'llmtrainer':
        trainer_builder = LLMTrainer
        return trainer_builder


register_trainer('llmtrainer', call_llm_trainer)
