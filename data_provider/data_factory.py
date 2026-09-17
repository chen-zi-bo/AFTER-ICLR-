from data_provider.data_loader import Dataset_TextTimeSeries, Dataset_GDELT
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'agriculture': Dataset_TextTimeSeries,
    'climate': Dataset_TextTimeSeries,
    'economy': Dataset_TextTimeSeries,
    'energy': Dataset_TextTimeSeries,
    'environment': Dataset_TextTimeSeries,
    'health': Dataset_TextTimeSeries,
    'security': Dataset_TextTimeSeries,
    'socialgood': Dataset_TextTimeSeries,
    'traffic': Dataset_TextTimeSeries,
    'weather': Dataset_TextTimeSeries,
    'gdelt': Dataset_GDELT,

}


def data_provider(args, flag, eventcode=1, generator=None):
    data_name = str(args.data).lower()
    if data_name not in data_dict:
        supported = ', '.join(sorted(data_dict.keys()))
        raise ValueError(f'Unknown dataset: {args.data}. Supported datasets: {supported}')

    # 选择对应的数据集类（自定义）
    Data = data_dict[data_name]
    # 时间日期戳编码
    timeenc = 0 if args.embed != 'timeF' else 1
    # 测试不打乱，其余打乱
    shuffle_flag = False if flag == 'test' else True

    # drop_last = flag == 'train'
    drop_last = True
    batch_size = args.batch_size
    freq = args.freq


    if data_name == 'gdelt':
        data_set = Data(
            args=args,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            channel_independent=False,
            event_root_code=eventcode
        )
    else:
        # 创建数据集，并传入对应参数
        data_set = Data(
            args=args,
            dataset_name=data_name,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            channel_independent=args.channel_independent
        )
    print(flag, len(data_set))
    # 创建数据loader
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        generator=generator)
    return data_set, data_loader
