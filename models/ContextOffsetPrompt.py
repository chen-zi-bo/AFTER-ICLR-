import json
import re
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

GDELT_VARIABLE_DESCRIPTION = (
    '`NumMentions` is the number of mentions, `NumArticles` is the number of related articles, '
    'and `NumSources` is the number of distinct sources.'
)

DATASET_CARDS = {
    'agriculture': {
        'dataset_description': (
            'Agriculture is a dataset on the U.S. broiler market, built from USDA ERS retail broiler '
            'composite prices and aligned USDA market reports/news snippets. It spans 1983-01 to '
            '2024-04 and is sampled monthly.'
        ),
        'variable_description': '`OT` is the retail broiler composite price series.',
        'context_guidance': (
            'Useful context should explain demand, supply, production cost, disease, trade, policy, '
            'inventory, or competing-meat pressure that can move broiler prices beyond the recent price path.'
        ),
    },
    'climate': {
        'dataset_description': (
            'Climate is a dataset on contiguous U.S. monthly precipitation, built from NOAA climate '
            'records and aligned NOAA precipitation/drought/climate reports/news snippets. It spans '
            '1983-01 to 2024-04 and is sampled monthly.'
        ),
        'variable_description': (
            '`OT` is the monthly contiguous U.S. precipitation amount. Higher values mean wetter '
            'conditions, while lower values mean drier conditions.'
        ),
        'context_guidance': (
            'Useful context should explain precipitation-relevant weather or climate conditions that can '
            'affect the forecast months. Drought or dryness implies lower precipitation only when it is '
            'active or persistent into the forecast window; wet storms or wet-season anomalies imply higher precipitation.'
        ),
    },
    'economy': {
        'dataset_description': (
            'Economy is a dataset on the U.S. international trade balance, built from U.S. Census '
            'foreign-trade data and aligned monthly economic reports/news snippets. It spans 1989-01 '
            'to 2024-03 and is sampled monthly.'
        ),
        'variable_description': '`OT` is the U.S. international trade balance, computed as Exports minus Imports.',
        'context_guidance': (
            'Useful context should explain export or import pressure. Stronger exports or weaker imports '
            'raise OT; weaker exports or stronger imports lower OT and can make the trade balance more negative.'
        ),
    },
    'energy': {
        'dataset_description': (
            'Energy is a dataset on U.S. gasoline prices, built from EIA gasoline price data and '
            'aligned energy-market reports/news snippets. It spans 1996-01 to 2024-04 and is sampled '
            'weekly.'
        ),
        'variable_description': '`OT` is the weekly U.S. retail gasoline price.',
        'context_guidance': (
            'Useful context should explain crude oil costs, refinery outages, inventories, taxes, seasonal demand, '
            'geopolitical supply risk, or demand destruction that can move gasoline prices beyond the recent price path.'
        ),
    },
    'environment': {
        'dataset_description': (
            'Environment is a dataset on New York air quality, built from EPA outdoor air-quality data '
            'and aligned local air-quality reports/news snippets. It spans 1982-01-01 to 2023-09-30 '
            'and is sampled daily.'
        ),
        'variable_description': '`OT` is the daily Air Quality Index for the New York metro area.',
        'context_guidance': (
            'Useful context should explain pollutant sources or dispersion conditions affecting New York metro AQI, '
            'such as locally relevant wildfire smoke, stagnant air, temperature inversions, strong winds, rain, '
            'emissions changes, or policy disruptions. Match location and timing carefully; short air-quality events '
            'normally support localized offsets rather than effects across a long forecast horizon.'
        ),
    },
    'health': {
        'dataset_description': (
            'Health is a Public Health (United States) dataset on influenza patients proportion, '
            'built from CDC ILINet surveillance data and aligned weekly influenza reports/news '
            'snippets. It spans 1997-09-29 to 2024-05-06 and is sampled weekly.'
        ),
        'variable_description': '`OT` is the weekly influenza patients proportion in the United States.',
        'context_guidance': (
            'Useful context should explain influenza transmission pressure such as outbreaks, strain severity, '
            'vaccination mismatch, school or holiday mixing, public-health interventions, or unusual season timing.'
        ),
    },
    'security': {
        'dataset_description': (
            'Security is a dataset on U.S. disaster and emergency grants, built from FEMA OpenFEMA '
            'data and aligned disaster and emergency declaration reports/news snippets. It spans '
            '1999-09-01 to 2024-05-01 and is sampled monthly.'
        ),
        'variable_description': '`OT` is the monthly FEMA disaster and emergency grant amount.',
        'context_guidance': (
            'Useful context must bear on the timing or amount of monthly FEMA grant obligations or recovery spending. '
            'A disaster report or emergency declaration alone does not determine when grants are obligated or their '
            'amount, so treat it as weak unless funding, obligation, scale, or a credible delayed spending mechanism '
            'connects it to the forecast months.'
        ),
    },
    'socialgood': {
        'dataset_description': (
            'SocialGood is a dataset on the U.S. labor market, built from BLS unemployment statistics '
            'and aligned employment situation/labor force reports/news snippets. It spans 1950-01-01 '
            'to 2024-04-01 and is sampled monthly.'
        ),
        'variable_description': '`OT` is the U.S. unemployment rate.',
        'context_guidance': (
            'Useful context should explain changes in the official aggregate U.S. unemployment rate through broad '
            'layoffs, hiring, recession, labor-force participation, or policy shocks. Distinguish national BLS '
            'unemployment from local, sector, demographic, alternative, and historical rates. Reports of already '
            'observed unemployment levels usually confirm the history rather than provide a new future driver.'
        ),
    },
    'traffic': {
        'dataset_description': (
            'Traffic is a dataset on U.S. travel demand, built from FHWA vehicle-miles-traveled data '
            'and aligned traffic volume reports/news snippets. It spans 1980-01-01 to 2024-03-01 and '
            'is sampled monthly.'
        ),
        'variable_description': '`OT` is the monthly U.S. travel volume measured by vehicle miles '
                                'traveled.',
        'context_guidance': (
            'Useful context should explain broad U.S. road vehicle-miles traveled through holidays, commuting changes, '
            'fuel prices, weather disruptions, mobility restrictions, economic activity, or infrastructure shocks. '
            'Do not treat local congestion, one road or state, air travel, airport traffic, or generic traffic-count '
            'definitions as direct evidence for national monthly VMT unless a credible aggregate mechanism is present.'
        ),
    },
    'weather': {
        'dataset_description': (
            'Weather is a weather dataset built from intraday meteorological observations and aligned '
            'natural-language weather descriptions. It spans 2012-07-17 06:00 to 2023-10-20 16:00 '
            'with three observations per day at irregularly spaced hours.'
        ),
        'variable_description': '`OT` is the maximum humidity in the local weather series.',
        'context_guidance': (
            'Useful context should explain local humidity drivers such as air mass changes, precipitation, fronts, '
            'temperature changes, wind shifts, or short-term weather systems. Use the actual timestamps to judge '
            'duration: descriptions of current observed weather are often already reflected in the numerical history '
            'and generally support, at most, the early part of a multi-day forecast unless persistence is explicit.'
        ),
    },
    'gdelt': {
        'dataset_description': (
            'GDELT is a news-event time-series dataset built from GDELT event records and summaries. '
            'It tracks media attention for event categories.'
        ),
        'variable_description': GDELT_VARIABLE_DESCRIPTION,
        'context_guidance': (
            'Useful context should explain concrete news developments likely to change event media attention, '
            'source diversity, or article volume beyond the recent event-count pattern.'
        ),
    },
}

GDELT_EVENT_CARDS = {
    1: 'EventRootCode 01 is a GDELT-based U.S. event series for public statements, built from GDELT news media event records and summaries. It spans 2022-08-18 to 2023-07-31 and is sampled daily.',
    2: 'EventRootCode 02 is a GDELT-based U.S. event series for appeals or calls for action, built from GDELT news media event records and summaries. It spans 2022-08-18 to 2023-07-31 and is sampled daily.',
    3: 'EventRootCode 03 is a GDELT-based U.S. event series for expressed cooperation intent, built from GDELT news media event records and summaries. It spans 2022-08-18 to 2023-07-31 and is sampled daily.',
    4: 'EventRootCode 04 is a GDELT-based U.S. event series for consultations, built from GDELT news media event records and summaries. It spans 2022-08-18 to 2023-07-31 and is sampled daily.',
    5: 'EventRootCode 05 is a GDELT-based U.S. event series for diplomatic cooperation, built from GDELT news media event records and summaries. It spans 2022-08-18 to 2023-07-31 and is sampled daily.',
    7: 'EventRootCode 07 is a GDELT-based U.S. event series for aid provision, built from GDELT news media event records and summaries. It spans 2022-08-18 to 2023-07-31 and is sampled daily.',
    8: 'EventRootCode 08 is a GDELT-based U.S. event series for yielding or concession behavior, built from GDELT news media event records and summaries. It spans 2022-08-18 to 2023-07-31 and is sampled daily.',
    11: 'EventRootCode 11 is a GDELT-based U.S. event series for disapproval events, built from GDELT news media event records and summaries. It spans 2022-08-18 to 2023-07-31 and is sampled daily.',
    17: 'EventRootCode 17 is a GDELT-based U.S. event series for coercive events, built from GDELT news media event records and summaries. It spans 2022-08-18 to 2023-07-31 and is sampled daily.',
    19: 'EventRootCode 19 is a GDELT-based U.S. event series for conflict or fight events, built from GDELT news media event records and summaries. It spans 2022-08-18 to 2023-07-31 and is sampled daily.',
}


def _is_missing_value(value: Any) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def context_json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _clean_text(value: Any, max_chars: int) -> str:
    if value is None or _is_missing_value(value):
        return ''
    text = str(value).strip()
    if not text or text.lower() in {'nan', 'none', 'na'}:
        return ''
    text = re.sub(r'\s+', ' ', text)
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rstrip() + ' ...'
    return text


class DFLoaderTextSerializer:
    """Serialize a DataFrame into DFLoader-style pandas code snippet text."""

    def __init__(self, precision: int = 4):
        self.precision = precision

    def serialize(self, df: pd.DataFrame, signed_numeric_columns: Sequence[str] = ()) -> str:
        if df.empty and len(df.columns) == 0:
            return 'pd.DataFrame({})'

        signed_columns = {str(column) for column in signed_numeric_columns}
        column_lines = []
        for column in df.columns:
            signed = str(column) in signed_columns
            values = [self._format_value(value, signed=signed) for value in df[column].tolist()]
            column_name = json.dumps(str(column), ensure_ascii=False)
            column_lines.append(f'    {column_name} : [{", ".join(values)}]')

        index_values = [self._format_value(value) for value in df.index.tolist()]
        index_text = ', '.join(index_values)
        if column_lines:
            body = ',\n'.join(column_lines)
            return f'pd.DataFrame({{\n{body}\n}},\nindex=[{index_text}])'
        return f'pd.DataFrame({{}},\nindex=[{index_text}])'

    def _format_value(self, value: Any, signed: bool = False) -> str:
        if value is None or _is_missing_value(value):
            return 'None'
        if isinstance(value, (bool, np.bool_)):
            return 'True' if bool(value) else 'False'
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return 'None'
            if signed:
                return f'{float(value):+.{self.precision}f}'
            return f'{float(value):.{self.precision}f}'
        if isinstance(value, (int, np.integer)):
            if signed:
                return f'{int(value):+d}'
            return str(int(value))
        if isinstance(value, (list, tuple, np.ndarray)):
            return '[' + ', '.join(self._format_value(item, signed=signed) for item in list(value)) + ']'
        text = re.sub(r'\s+', ' ', str(value)).strip()
        return json.dumps(text, ensure_ascii=False)


class ContextOffsetPromptAssembler:
    def __init__(self, configs, settings: Dict[str, Any]):
        self.configs = configs
        self.settings = settings
        self.serializer = DFLoaderTextSerializer(
            precision=int(self._setting('context_table_precision', 'table_precision'))
        )
        self.stage1_template = self._read_text(
            self._setting('context_stage1_template_path', 'stage1_template_path')
        )
        self.stage2_template = self._read_text(
            self._setting('context_stage2_template_path', 'stage2_template_path')
        )
        self.stage2_memory_template = self._read_text(
            self._setting('context_stage2_memory_template_path', 'stage2_memory_template_path')
        )
        self.stage1_output_template = self._read_text(
            self._setting('context_stage1_output_template_path', 'stage1_output_template_path')
        )

    def build_stage1_prompt(
            self,
            history_values: np.ndarray,
            history_times: Sequence[str],
            future_times: Sequence[str],
            context_texts: Sequence[str],
            variables: Sequence[str],
    ) -> str:
        history_values = self._ensure_2d(np.asarray(history_values, dtype=float))
        variables = self._normalize_variables(variables, history_values.shape[-1])
        initial = history_values[0]
        diff_values = history_values - initial
        time_series_df = pd.DataFrame(diff_values, columns=variables)
        time_series_df.insert(0, 'time', list(history_times))

        # 获得time和文本配对
        context_rows = self._context_rows(history_times, context_texts)
        context_df = pd.DataFrame(context_rows, columns=['time', 'text'])

        # 插入prompt
        mapping = self._base_mapping(
            variables=variables,
            history_times=history_times,
            history_values=history_values,
        )
        mapping.update({
            '{input_time_range}': self._time_range_text(history_times),
            '{forecast_time_range}': self._time_range_text(future_times),
            '{time_series_dfloader_text}': self.serializer.serialize(
                time_series_df,
                signed_numeric_columns=variables,
            ),
            '{context_texts_dfloader_text}': self.serializer.serialize(context_df),
            '{template}': self.stage1_output_template.strip(),
        })
        return self._replace_many(self.stage1_template, mapping)

    def build_stage2_prompt(
            self,
            history_values: np.ndarray,
            history_times: Sequence[str],
            future_times: Sequence[str],
            base_forecast: np.ndarray,
            variables: Sequence[str],
            stage1_json: Dict[str, Any],
    ) -> str:
        return self._build_stage2_prompt(
            template=self.stage2_template,
            history_values=history_values,
            history_times=history_times,
            future_times=future_times,
            base_forecast=base_forecast,
            variables=variables,
            stage1_json=stage1_json,
        )

    def build_stage2_memory_prompt(
            self,
            history_values: np.ndarray,
            history_times: Sequence[str],
            future_times: Sequence[str],
            base_forecast: np.ndarray,
            variables: Sequence[str],
            stage1_json: Dict[str, Any],
            memory_examples: Sequence[Dict[str, Any]],
    ) -> str:
        memory_mapping = {
            '{retrieved_memory_cases}': self._format_memory_examples(
                memory_examples=memory_examples,
                variables=variables,
            ),
        }
        return self._build_stage2_prompt(
            template=self.stage2_memory_template,
            history_values=history_values,
            history_times=history_times,
            future_times=future_times,
            base_forecast=base_forecast,
            variables=variables,
            stage1_json=stage1_json,
            extra_mapping=memory_mapping,
        )

    def _format_memory_examples(
            self,
            memory_examples: Sequence[Dict[str, Any]],
            variables: Sequence[str],
    ) -> str:
        blocks = []
        for rank, memory_example in enumerate(memory_examples, start=1):
            entry = memory_example['entry']
            memory_history = self._ensure_2d(np.asarray(entry['history_values'], dtype=float))
            memory_true = self._ensure_2d(np.asarray(entry['true_future'], dtype=float))
            memory_variables = self._normalize_variables(variables, memory_history.shape[-1])
            forecast_variables = self._normalize_variables(memory_variables, memory_true.shape[-1])
            reference = memory_history[0]

            history_df = pd.DataFrame(memory_history - reference, columns=memory_variables)
            history_df.insert(0, 'time', list(entry['history_times']))
            true_df = pd.DataFrame(
                memory_true - reference[:memory_true.shape[-1]],
                columns=forecast_variables,
            )
            true_df.insert(0, 'time', list(entry['future_times']))
            reference_values = ', '.join(
                f'{var}={reference[idx]:.{self.serializer.precision}f}'
                for idx, var in enumerate(memory_variables)
            )
            stage1_state = entry.get('stage1_json', {})
            context_state = json.dumps(
                {
                    key: stage1_state.get(key)
                    for key in (
                        'context_status',
                        'history_state_summary',
                        'context_summary',
                    )
                    if stage1_state.get(key) not in (None, '', [])
                },
                ensure_ascii=False,
                default=context_json_default,
            )
            blocks.append('\n'.join([
                f'[Retrieved Historical Case {rank}]',
                f'input_time_range: {self._time_range_text(entry["history_times"])}',
                f'forecast_time_range: {self._time_range_text(entry["future_times"])}',
                'The history and observed true future use this case\'s historical initial value as their shared reference.',
                f'reference_values at {entry["history_times"][0]}: {reference_values}',
                '',
                '[Memory History Time Series]',
                self.serializer.serialize(history_df, signed_numeric_columns=memory_variables),
                '',
                '[Memory History-Aware Context State]',
                context_state,
                '',
                '[Memory Observed True Future]',
                self.serializer.serialize(true_df, signed_numeric_columns=forecast_variables),
            ]))
        return '\n\n'.join(blocks)

    def _build_stage2_prompt(
            self,
            template: str,
            history_values: np.ndarray,
            history_times: Sequence[str],
            future_times: Sequence[str],
            base_forecast: np.ndarray,
            variables: Sequence[str],
            stage1_json: Dict[str, Any],
            extra_mapping: Dict[str, str] = None,
    ) -> str:
        history_values = self._ensure_2d(np.asarray(history_values, dtype=float))
        base_forecast = self._ensure_2d(np.asarray(base_forecast, dtype=float))
        variables = self._normalize_variables(variables, history_values.shape[-1])
        forecast_variables = self._normalize_variables(variables, base_forecast.shape[-1])
        initial = history_values[0]
        diff_values = history_values - initial
        base_reference = initial
        base_reference = np.asarray(base_reference, dtype=float)
        if base_reference.ndim == 0:
            base_reference = np.repeat(base_reference.item(), len(forecast_variables))
        base_diff_values = base_forecast - base_reference

        time_series_df = pd.DataFrame(diff_values, columns=variables)
        time_series_df.insert(0, 'time', list(history_times))

        base_df = pd.DataFrame(base_diff_values, columns=forecast_variables)
        base_df.insert(0, 'time', list(future_times))

        base_time_0 = str(history_times[0]) if history_times else ''
        base_initial_values_text = ', '.join(
            f'{var}={base_reference[idx]:.{self.serializer.precision}f}' for idx, var in enumerate(forecast_variables)
        )

        mapping = self._base_mapping(
            variables=variables,
            history_times=history_times,
            history_values=history_values,
        )
        mapping.update({
            '{input_time_range}': self._time_range_text(history_times),
            '{forecast_time_range}': self._time_range_text(future_times),
            '{time_series_dfloader_text}': self.serializer.serialize(
                time_series_df,
                signed_numeric_columns=variables,
            ),
            '{base_forecast_dfloader_text}': self.serializer.serialize(
                base_df,
                signed_numeric_columns=forecast_variables,
            ),
            '{base_time_0}': base_time_0,
            '{base_initial_values_text}': base_initial_values_text,
            '{history_aware_context_summary}': json.dumps(
                stage1_json,
                ensure_ascii=False,
                default=context_json_default,
            ),
        })
        if extra_mapping:
            mapping.update(extra_mapping)
        return self._replace_many(template, mapping)

    @staticmethod
    def _ensure_2d(values: np.ndarray) -> np.ndarray:
        if values.ndim == 0:
            return values.reshape(1, 1)
        if values.ndim == 1:
            return values.reshape(-1, 1)
        return values

    def scale_hint_df(self, history_values: np.ndarray, variables: Sequence[str]) -> pd.DataFrame:
        history_values = self._ensure_2d(np.asarray(history_values, dtype=float))
        variables = self._normalize_variables(variables, history_values.shape[-1])
        if len(history_values) <= 1:
            diffs = np.zeros((1, len(variables)), dtype=float)
        else:
            diffs = np.diff(history_values, axis=0)
        rows = []
        for idx, variable in enumerate(variables):
            col = diffs[:, idx]
            rows.append({
                'variable': variable,
                'recent_mean_abs_change': float(np.mean(np.abs(col))),
                'recent_std_change': float(np.std(col)),
                'recent_max_abs_change': float(np.max(np.abs(col))),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _time_range_text(times: Sequence[str]) -> str:
        values = [str(time_value) for time_value in times]
        if not values:
            return ''
        if len(values) == 1:
            return values[0]
        return f'{values[0]} to {values[-1]}'

    def _normalize_variables(self, variables: Sequence[str], num_vars: int) -> List[str]:
        names = [str(variable) for variable in variables] if variables else []
        if len(names) >= num_vars:
            return names[:num_vars]
        names.extend(f'var_{idx}' for idx in range(len(names), num_vars))
        return names

    def _context_rows(self, history_times: Sequence[str], context_texts: Sequence[str]) -> List[Tuple[str, str]]:
        max_chars = int(self._setting('context_max_text_chars', 'max_text_chars'))
        include_empty = bool(int(self._setting('context_include_empty_texts', 'include_empty_texts')))
        rows = []
        for time_value, text_value in zip(history_times, context_texts):
            text = _clean_text(text_value, max_chars=max_chars)
            if text or include_empty:
                rows.append((time_value, text if text else 'No information available.'))
        return rows

    def _base_mapping(
            self,
            variables: Sequence[str],
            history_times: Sequence[str],
            history_values: np.ndarray,
    ) -> Dict[str, str]:
        dataset_key = str(getattr(self.configs, 'data', '')).lower()
        card = DATASET_CARDS.get(dataset_key, {})
        dataset_description = getattr(self.configs, 'context_dataset_description', None) or card.get(
            'dataset_description', f'{dataset_key} time-series dataset.'
        )
        variable_description = getattr(self.configs, 'context_variable_description', None) or card.get(
            'variable_description', ', '.join(variables)
        )
        context_guidance = getattr(self.configs, 'context_dataset_guidance', None) or card.get(
            'context_guidance',
            'Useful context should explain concrete drivers that can move the target variable during the forecast window.',
        )
        if dataset_key == 'gdelt':
            event_code = getattr(self.configs, 'eventcode', getattr(self.configs, 'event_root_code', None))
            try:
                event_code = None if event_code is None else int(event_code)
            except (TypeError, ValueError):
                event_code = None
            if event_code is not None:
                event_description = GDELT_EVENT_CARDS.get(event_code)
                if event_description is not None:
                    dataset_description = event_description
                variable_description = GDELT_VARIABLE_DESCRIPTION
                context_guidance = DATASET_CARDS['gdelt']['context_guidance']
        target_variable = ', '.join(variables) if variables else 'OT'
        base_backbone_name = getattr(
            self.configs,
            'context_base_model',
            getattr(self.configs, 'base_model', 'DLinear'),
        )
        frequency = getattr(self.configs, 'dataset_freq', '')
        time_0 = history_times[0] if history_times else '{time_0}'
        precision = int(self.serializer.precision)
        initial_values_text = ', '.join(
            f'{var}={history_values[0, idx]:.{precision}f}' for idx, var in enumerate(variables)
        )
        return {
            '{dataset_description}': str(dataset_description),
            '{context_guidance}': str(context_guidance),
            '{target_variable}': str(target_variable),
            '{base_backbone_name}': str(base_backbone_name),
            '{input_length}': str(getattr(self.configs, 'seq_len', len(history_times))),
            '{prediction_length}': str(getattr(self.configs, 'pred_len', '')),
            '{frequency}': str(frequency),
            '{variable_description}': str(variable_description),
            '{time_0}': str(time_0),
            '{initial_values_text}': initial_values_text,
        }

    def _setting(self, attr_name: str, setting_key: str) -> Any:
        return getattr(self.configs, attr_name, self.settings[setting_key])

    def _read_text(self, path: str) -> str:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def _replace_many(self, template: str, mapping: Dict[str, str]) -> str:
        output = template
        for key, value in mapping.items():
            output = output.replace(key, value)
        return output
