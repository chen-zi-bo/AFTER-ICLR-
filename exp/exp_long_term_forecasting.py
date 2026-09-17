import json
import logging
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import EarlyStopping, extract_model_state, load_checkpoint

warnings.filterwarnings('ignore')


dim_dict = {
    'agriculture': 1,
    'climate': 1,
    'economy': 1,
    'energy': 1,
    'environment': 1,
    'health': 1,
    'security': 1,
    'socialgood': 1,
    'traffic': 1,
    'weather': 1,
    'gdelt': 3
}

def save_attn_heatmap(attn_weights, save_path, feature_names=None):
    attn_mean = attn_weights.mean(dim=0)
    attn_np = attn_mean.detach().cpu().numpy()
    attn_np = attn_np / (attn_np.sum(axis=1, keepdims=True) + 1e-8)

    plt.figure(figsize=(16, 6))
    sns.heatmap(
        attn_np,
        cmap='Blues',
        yticklabels=feature_names if feature_names is not None else True,
        cbar=False
    )
    plt.xlabel('Text Position (L)')
    plt.ylabel('Feature (N)')
    plt.title('Feature-Text Attention')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def setup_logger(log_path):
    logger = logging.getLogger(log_path)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

        self._adjust_args_by_dataset()
        self.model = self._build_model().to(self.device)

        os.makedirs(self._experiment_dir(), exist_ok=True)
        self.logger = setup_logger(os.path.join(self._experiment_dir(), 'output.log'))

    def _experiment_dir(self):
        model_detail = (
            self.args.context_base_model
            if self.args.model == 'AFTER'
            else self.args.model
        )
        return os.path.join(
            self.args.checkpoints,
            f"{self.args.model}_{model_detail}_{self.args.data}_"
            f"{self.args.seed}_{self.args.seq_len}_{self.args.label_len}_{self.args.pred_len}_{self.args.eventcode}"
        )

    def _resume_checkpoint_path(self):
        return os.path.join(self._experiment_dir(), 'resume.pth')

    def _best_checkpoint_path(self):
        return os.path.join(self._experiment_dir(), 'checkpoint.pth')

    def _context_artifact_dir(self):
        folder_name = str(getattr(self.args, 'model_id', '')).strip()
        if not folder_name:
            folder_name = f'Test{self.args.data}_s{self.args.seed}_pl{self.args.pred_len}'
            event_code = getattr(self.args, 'eventcode', None)
            if event_code not in {None, '', 0, '0'}:
                folder_name += f'_e{event_code}'
        llm_name = self._context_llm_artifact_name()
        if llm_name:
            folder_name = f'{folder_name}_{llm_name}'
        if bool(int(getattr(self.args, 'context_memory_enabled', 0))):
            memory_name = str(getattr(
                self.args, 'context_memory_name', 'event_response_memory_v1'))
            memory_name = ''.join(
                char if char.isalnum() or char in {'-', '_', '.'} else '_'
                for char in memory_name
            ).strip('_')
            folder_name = f'{folder_name}_memory-{memory_name or "enabled"}'
        return os.path.join(
            getattr(self.args, 'context_output_dir', './context_offset_results'),
            folder_name,
        )

    def _context_llm_artifact_name(self):
        if self.args.model != 'AFTER':
            return ''
        backend = str(getattr(self.args, 'context_llm_backend', 'hf')).strip()
        if backend in {'deepseek', 'openai_compatible'}:
            model_name = getattr(self.args, 'context_api_model', '')
        else:
            model_name = getattr(self.args, 'context_llm_model_name_or_path', '')
        label = f'{backend}_{model_name}' if model_name else backend
        label = os.path.basename(str(label).rstrip('/\\'))
        safe_chars = []
        for char in label:
            if char.isalnum() or char in {'-', '_', '.'}:
                safe_chars.append(char)
            else:
                safe_chars.append('_')
        return ''.join(safe_chars).strip('_')

    def _adjust_args_by_dataset(self):
        data = self.args.data.lower()
        if data not in dim_dict:
            raise ValueError(f'Unknown dataset: {data}')

        effective_model = (
            getattr(self.args, 'context_base_model', self.args.model)
            if self.args.model == 'AFTER'
            else self.args.model
        )
        dim = dim_dict[data]
        self.args.enc_in = dim
        self.args.dec_in = dim
        self.args.c_out = dim

        if self.args.channel_independent:
            self.args.enc_in = 1
            self.args.dec_in = 1
            self.args.c_out = 1

    def _build_model(self):
        model_module = self.get_model_module(self.args.model)
        if self.args.model == 'AFTER':
            model = model_module.Model(self.args, self.device).float()
        else:
            model = model_module.Model(self.args).float()
        return model

    def _get_data(self, flag, generator=None):
        if self.args.data == 'gdelt':
            return data_provider(self.args, flag, self.args.eventcode, generator=generator)
        return data_provider(self.args, flag, generator=generator)

    def _create_train_loader(self, train_data, epoch):
        generator = torch.Generator()
        generator.manual_seed(self.args.seed + epoch)
        return DataLoader(
            train_data,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            drop_last=True,
            generator=generator
        )

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def _forward_model(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        if self.args.isUni:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            return outputs

        if self.args.output_attention:
            return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    def _prepare_batch(self, dataset, batch_x, batch_y, batch_x_mark, batch_y_mark, index):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        outputs = self._forward_model(batch_x, batch_y, batch_x_mark, batch_y_mark)
        outputs = outputs[:, -self.args.pred_len:, :]
        batch_y = batch_y[:, -self.args.pred_len:, :]
        return outputs, batch_y, batch_x, None

    def _move_optimizer_state_to_device(self, optimizer):
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

    def _load_model_from_path(self, checkpoint_path):
        checkpoint = load_checkpoint(checkpoint_path, map_location='cpu')
        state_dict = extract_model_state(checkpoint)
        if state_dict is not None:
            self.model.load_state_dict(state_dict)
        return checkpoint

    def _inverse_numpy(self, dataset, values):
        values = np.asarray(values)
        if not getattr(dataset, 'scale', False):
            return values
        shape = values.shape
        flat_values = values.reshape(-1, shape[-1])
        inverted = dataset.inverse_transform(flat_values)
        return inverted.reshape(shape)

    def _date_value_to_str(self, value):
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                return ''
            value = value[0]
        try:
            stamp = np.datetime64(value)
            text = np.datetime_as_string(stamp, unit='s')
            return text.replace('T00:00:00', '').replace('T', ' ')
        except Exception:
            return str(value)

    def _date_matrix_to_strings(self, dates):
        dates = np.asarray(dates)
        return [[self._date_value_to_str(value) for value in row] for row in dates]

    def _get_context_times(self, dataset, index, hist_len, pred_len):
        if hasattr(dataset, 'get_date'):
            history_dates = dataset.get_date(index)
            if isinstance(history_dates, tuple):
                history_dates = history_dates[-1]
            history_times = self._date_matrix_to_strings(history_dates)
        else:
            batch_size = len(index) if hasattr(index, '__len__') else 1
            history_times = [[f't{step}' for step in range(hist_len)] for _ in range(batch_size)]

        if hasattr(dataset, 'get_forecast_date'):
            future_dates = dataset.get_forecast_date(index)
            if isinstance(future_dates, tuple):
                future_dates = future_dates[-1]
            future_times = self._date_matrix_to_strings(future_dates)
        else:
            batch_size = len(index) if hasattr(index, '__len__') else 1
            future_times = [[f't+{step + 1}' for step in range(pred_len)] for _ in range(batch_size)]
        return history_times, future_times

    def _index_to_list(self, index):
        if isinstance(index, torch.Tensor):
            return index.detach().cpu().numpy().tolist()
        if isinstance(index, np.ndarray):
            return index.tolist()
        if isinstance(index, (list, tuple)):
            return list(index)
        return [index]

    def _save_resume_state(self, model_optim, scaler, epoch, step_in_epoch, early_stopping, finished=False):
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': model_optim.state_dict(),
            'epoch': epoch,
            'step_in_epoch': step_in_epoch,
            'finished': finished,
            'early_stopping': early_stopping.state_dict(),
        }
        if scaler is not None:
            state['scaler_state_dict'] = scaler.state_dict()
        torch.save(state, self._resume_checkpoint_path())

    def _load_resume_state(self, model_optim, scaler, early_stopping):
        if not getattr(self.args, 'resume', 1):
            return 0, 0, False

        resume_state = load_checkpoint(self._resume_checkpoint_path(), map_location='cpu')
        if resume_state is None:
            return 0, 0, False

        state_dict = extract_model_state(resume_state)
        if state_dict is not None:
            self.model.load_state_dict(state_dict)
        if 'optimizer_state_dict' in resume_state:
            model_optim.load_state_dict(resume_state['optimizer_state_dict'])
            self._move_optimizer_state_to_device(model_optim)
        if scaler is not None and 'scaler_state_dict' in resume_state:
            scaler.load_state_dict(resume_state['scaler_state_dict'])
        early_stopping.load_state_dict(resume_state.get('early_stopping'))

        start_epoch = int(resume_state.get('epoch', 0))
        start_step = int(resume_state.get('step_in_epoch', 0))
        finished = bool(resume_state.get('finished', False))
        self.logger.info(
            'Auto resume state found: epoch=%s, step_in_epoch=%s, finished=%s',
            start_epoch, start_step, finished
        )
        return start_epoch, start_step, finished

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark, index in vali_loader:
                outputs, target, _, _ = self._prepare_batch(
                    vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark, index
                )
                loss = criterion(outputs.detach().cpu(), target.detach().cpu())
                total_loss.append(loss.item())
        self.model.train()
        return np.average(total_loss) if total_loss else np.inf

    def train(self, setting):
        train_data, _ = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        os.makedirs(self._experiment_dir(), exist_ok=True)

        time_now = time.time()
        train_steps = len(self._create_train_loader(train_data, 0))
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        start_epoch, start_step, finished = self._load_resume_state(model_optim, scaler, early_stopping)
        if finished:
            self.logger.info('Experiment already finished before. Loading best checkpoint directly.')
            self._load_model_from_path(self._best_checkpoint_path())
            return self.model

        for epoch in range(start_epoch, self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            train_loader = self._create_train_loader(train_data, epoch)
            skip_steps = start_step if epoch == start_epoch else 0

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(train_loader):
                if i < skip_steps:
                    continue

                iter_count += 1
                model_optim.zero_grad()
                outputs, target, _, _ = self._prepare_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark, index
                )
                loss = criterion(outputs, target)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    self.logger.info('\titers: %s, epoch: %s | loss: %.7f', i + 1, epoch + 1, loss.item())
                    speed = (time.time() - time_now) / max(iter_count, 1)
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    self.logger.info('\tspeed: %.4fs/iter; left time: %.4fs', speed, left_time)
                    iter_count = 0
                    time_now = time.time()

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if getattr(self.args, 'resume', 1):
                    self._save_resume_state(model_optim, scaler, epoch, i + 1, early_stopping, finished=False)

            self.logger.info('Epoch: %s cost time: %s', epoch + 1, time.time() - epoch_time)
            train_loss = np.average(train_loss) if train_loss else np.inf
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            self.logger.info(
                'Epoch: %s, Steps: %s | Train Loss: %.7f Vali Loss: %.7f Test Loss: %.7f',
                epoch + 1, train_steps, train_loss, vali_loss, test_loss
            )

            best_extra_state = {
                'epoch': epoch + 1,
                'optimizer_state_dict': model_optim.state_dict(),
                'early_stopping': early_stopping.state_dict(),
            }
            if scaler is not None:
                best_extra_state['scaler_state_dict'] = scaler.state_dict()
            early_stopping(vali_loss, self.model, self._experiment_dir(), extra_state=best_extra_state)

            self._save_resume_state(model_optim, scaler, epoch + 1, 0, early_stopping, finished=False)
            if early_stopping.early_stop:
                self.logger.info('Early stopping')
                break

        self._load_model_from_path(self._best_checkpoint_path())
        self._save_resume_state(model_optim, scaler, self.args.train_epochs, 0, early_stopping, finished=True)
        return self.model

    def _build_context_memory_if_needed(self):
        if self.args.model != 'AFTER' or not bool(
                int(getattr(self.args, 'context_memory_enabled', 0))):
            return

        memory_path = self.model.context_memory_path()
        rebuild = bool(int(getattr(self.args, 'context_memory_rebuild', 0)))
        if os.path.isfile(memory_path) and not rebuild:
            self.logger.info('Reusing ContextOffset memory: %s', memory_path)
            return

        histories = []
        true_futures = []
        all_history_times = []
        all_future_times = []
        all_context_texts = []
        all_indices = []
        all_source_splits = []

        source_split = 'train'
        source_data, _ = self._get_data(flag=source_split)
        variables = getattr(source_data, 'variable_names', [self.args.target])
        memory_loader = DataLoader(
            source_data,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=False,
        )
        self.logger.info(
            'Collecting every deterministic train window for ContextOffset continuation memory.'
        )
        for batch_x, batch_y, _, _, index in memory_loader:
            target = batch_y[:, -self.args.pred_len:, :].float()
            index_list = self._index_to_list(index)
            history_times, future_times = self._get_context_times(
                source_data, index, batch_x.shape[1], target.shape[1]
            )
            context_texts = (
                source_data.get_text(index)
                if hasattr(source_data, 'get_text')
                else [[] for _ in index_list]
            )
            histories.append(batch_x.float().numpy())
            true_futures.append(target.numpy())
            all_history_times.extend(history_times)
            all_future_times.extend(future_times)
            all_context_texts.extend(context_texts)
            all_indices.extend(index_list)
            all_source_splits.extend([source_split] * len(index_list))

        summary = self.model.build_context_memory(
            history_values=np.concatenate(histories, axis=0),
            history_times=all_history_times,
            context_texts=all_context_texts,
            future_times=all_future_times,
            true_future=np.concatenate(true_futures, axis=0),
            variables=variables,
            sample_indices=all_indices,
            source_splits=all_source_splits,
        )
        self.logger.info('ContextOffset memory ready: %s', summary)

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            self.logger.info('loading model')
            self._load_model_from_path(self._best_checkpoint_path())
        self._build_context_memory_if_needed()

        preds = []
        trues = []
        inputs = []

        context_enabled = self.args.model == 'AFTER'
        context_base_preds = []
        context_offsets = []
        context_records = []

        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark, index in test_loader:
                outputs, target_tensor, batch_x_device, _ = self._prepare_batch(
                    test_data, batch_x, batch_y, batch_x_mark, batch_y_mark, index
                )
                outputs = outputs.detach().cpu().numpy()
                target = target_tensor.detach().cpu().numpy()
                batch_input_np = batch_x_device.detach().cpu().numpy()

                if context_enabled:
                    base_scaled = outputs
                    history_scaled = batch_input_np
                    history_times, future_times = self._get_context_times(
                        test_data, index, history_scaled.shape[1], base_scaled.shape[1]
                    )
                    batch_text = test_data.get_text(index) if hasattr(test_data, 'get_text') else [[] for _ in range(base_scaled.shape[0])]
                    variables = getattr(test_data, 'variable_names', [self.args.target])

                    adjusted, offsets, records = self.model.generate_context_adjusted_forecast(
                        history_values=history_scaled,
                        history_times=history_times,
                        context_texts=batch_text,
                        future_times=future_times,
                        base_forecast=base_scaled,
                        variables=variables,
                        sample_indices=self._index_to_list(index),
                    )
                    outputs = adjusted
                    context_base_preds.append(base_scaled)
                    context_offsets.append(offsets)
                    context_records.extend(records)

                if not context_enabled and test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = self._inverse_numpy(test_data, outputs).reshape(shape)
                    target = self._inverse_numpy(test_data, target).reshape(shape)

                preds.append(outputs[:, :, :])
                trues.append(target[:, :, :])
                inputs.append(batch_input_np)

        if not preds:
            raise RuntimeError('Test loader produced no batches.')

        # Test batches can have different batch sizes because drop_last=False for test.
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputs = np.concatenate(inputs, axis=0)

        self.logger.info('test shape: %s %s', preds.shape, trues.shape)

        result_folder = './results/' + setting + '/'
        os.makedirs(result_folder, exist_ok=True)

        dtw = -999
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        self.logger.info('mse:%s, mae:%s, dtw:%s', mse, mae, dtw)

        if context_enabled:
            context_folder = self._context_artifact_dir()
            base_preds = np.concatenate(context_base_preds, axis=0)
            offsets = np.concatenate(context_offsets, axis=0)
            self.model.save_context_artifacts(
                context_folder,
                context_records,
                inputs=inputs,
                base_preds=base_preds,
                offsets=offsets,
                preds=preds,
                trues=trues,
            )
            with open(os.path.join(context_folder, 'metrics.json'), 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        'mae': mae,
                        'mse': mse,
                        'rmse': rmse,
                        'mape': mape,
                        'mspe': mspe,
                        'base_pred_file': 'base_pred.npy',
                        'offset_file': 'offset.npy',
                        'pred_file': 'pred.npy',
                        'true_file': 'true.npy',
                    },
                    f,
                    indent=2,
                )

        with open(self.args.save_name, 'a') as f:
            backbone_name = self.args.context_base_model if context_enabled else self.args.model
            f.write(
                f"{self.args.model}_{backbone_name}_{self.args.data}_"
                f"{self.args.seed}_{self.args.seq_len}_{self.args.label_len}_{self.args.pred_len}_{self.args.eventcode}  \n"
            )
            f.write(f'mse:{mse}, mae:{mae}, rmse:{rmse}, mape:{mape}, mspe:{mspe}')
            f.write('\n\n')

        np.save(result_folder + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(result_folder + 'pred.npy', preds)
        np.save(result_folder + 'true.npy', trues)

        return mse
