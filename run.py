import argparse
import os
import random

import numpy as np
import torch

from utils.print_args import print_args

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DATASET_FREQ_MAP = {
    'agriculture': 'monthly',
    'climate': 'monthly',
    'economy': 'monthly',
    'energy': 'weekly',
    'environment': 'daily',
    'health': 'weekly',
    'security': 'monthly',
    'socialgood': 'monthly',
    'traffic': 'monthly',
    'weather': 'hourly',
    'gdelt': 'daily',
}
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='AFTER multimodal time-series forecasting')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='AFTER',
                        help='AFTER or a supported numerical time-series model')
    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (percent)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--channel_independent', type=int, default=0,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=3, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default='avg',
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=48,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--resume', type=int, default=1, help='auto resume an interrupted experiment')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--gpu', type=int, default=0, help='gpu')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--save_name', type=str, default='result_longterm_forecast', help='save name')
    parser.add_argument('--eventcode', type=int, default=1)

    # AFTER (legacy implementation name: ContextOffset)
    parser.add_argument('--context_base_model', type=str, default='DLinear',
                        help='base time-series backbone used inside AFTER')
    parser.add_argument('--context_llm_backend', type=str, default='deepseek',
                        choices=['hf', 'deepseek', 'openai_compatible'],
                        help='reasoning backend for AFTER prompts')
    parser.add_argument('--context_llm_model_name_or_path', type=str, default='./llm/Qwen3-8B',
                        help='local path or HuggingFace model id for the instruction LLM')
    parser.add_argument('--context_llm_device_map', type=str, default='auto')
    parser.add_argument('--context_llm_dtype', type=str, default='float16',
                        help='float16, bfloat16, or float32')

    parser.add_argument('--context_llm_temperature', type=float, default=0.0)
    parser.add_argument('--context_llm_top_p', type=float, default=1.0)

    parser.add_argument('--context_llm_trust_remote_code', type=int, default=1)
    # 是否只读本地
    parser.add_argument('--context_llm_local_files_only', type=int, default=1)
    parser.add_argument('--context_llm_load_in_4bit', type=int, default=0)
    parser.add_argument('--context_llm_batch_size', type=int, default=1,
                        help='local HF generation batch size; API backends ignore this option')
    parser.add_argument('--context_llm_max_input_tokens', type=int, default=0,
                        help='optional left-truncated input limit for local HF generation')
    parser.add_argument('--context_stage1_max_new_tokens', type=int, default=2048)
    parser.add_argument('--context_stage2_max_new_tokens', type=int, default=2048)

    parser.add_argument('--context_api_base_url', type=str, default='https://api.deepseek.com',
                        help='OpenAI-compatible base URL for API reasoning backends')
    parser.add_argument('--context_api_model', type=str, default='deepseek-v4-flash',
                        help='model name for API reasoning backends')
    parser.add_argument('--context_api_key_env', type=str, default='DEEPSEEK_API_KEY',
                        help='environment variable containing the API key')
    parser.add_argument('--context_api_key', type=str, default=None,
                        help='API key value; prefer context_api_key_env for experiments')
    parser.add_argument('--context_api_timeout', type=float, default=120.0)
    parser.add_argument('--context_api_max_retries', type=int, default=3)
    parser.add_argument('--context_api_retry_wait', type=float, default=2.0)
    parser.add_argument('--context_api_concurrency', type=int, default=None,
                        help='max concurrent per-sample API calls for AFTER API backends')
    parser.add_argument('--context_api_response_format', type=str, default='json_object',
                        choices=['none', 'json_object'])
    parser.add_argument('--context_api_top_k', type=int, default=None,
                        help='optional top-k sampling parameter for OpenAI-compatible local servers')
    parser.add_argument('--context_api_min_p', type=float, default=None,
                        help='optional min-p sampling parameter for OpenAI-compatible local servers')
    parser.add_argument('--context_api_enable_thinking', type=int, default=None, choices=[0, 1],
                        help='Qwen3 chat-template hard switch for OpenAI-compatible local servers')
    parser.add_argument('--context_api_thinking_type', type=str, default='auto',
                        choices=['auto', 'none', 'enabled', 'disabled'])
    parser.add_argument('--context_api_reasoning_effort', type=str, default='high')

    parser.add_argument('--context_output_dir', type=str, default='./context_offset_results')
    parser.add_argument('--context_memory_enabled', type=int, default=0,
                        help='build/load train event-response memory for AFTER')
    parser.add_argument('--context_memory_name', type=str, default='continuation_state_memory_v1')
    parser.add_argument('--context_memory_rebuild', type=int, default=0,
                        help='rebuild AFTER memory even when its JSONL file exists')

    parser.add_argument('--dataset_freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

    # Timemixer
    parser.add_argument('--timemixer_channel', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    # uni
    parser.add_argument('--isUni', default=False)

    args = parser.parse_args()
    dataset_key = str(args.data).lower()
    if dataset_key in DATASET_FREQ_MAP:
        args.dataset_freq = DATASET_FREQ_MAP[dataset_key]
    # 参数设置
    if args.model == "LightTS":
        if args.pred_len < args.seq_len:
            args.seq_len = args.pred_len
    effective_model = args.context_base_model if args.model == 'AFTER' else args.model
    if effective_model == 'TimeMixer' and args.data == 'gdelt':
        args.down_sampling_layers = 1
        args.down_sampling_window = 1
    if args.task_name == 'classification':
        args.batch_size = 16
        args.train_epochs = 100
    fix_seed = args.seed
    print("Now using seed {}".format(fix_seed))
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    args.use_gpu = True if torch.cuda.is_available() else False

    print(torch.cuda.is_available())

    print('Args in experiment:')
    print_args(args)
    # 根据任务名称实例化对应对象
    if args.task_name == 'long_term_forecast':
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        raise NotImplementedError('Imputation experiment module is not available in the current workspace.')
    elif args.task_name == 'anomaly_detection':
        raise NotImplementedError('Anomaly detection experiment module is not available in the current workspace.')
    elif args.task_name == 'classification':
        raise NotImplementedError('Classification experiment module is not available in the current workspace.')
    else:
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        Exp = Exp_Long_Term_Forecast
    # 根据train还是test调用对应对象的方法
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            now_mse = exp.test(setting, test=1)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        now_mse = exp.test(setting, test=1)

        torch.cuda.empty_cache()
