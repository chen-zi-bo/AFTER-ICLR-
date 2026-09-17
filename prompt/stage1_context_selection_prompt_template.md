``You are an expert assistant for history-aware contextual analysis in multimodal time-series forecasting.
Your job is to synthesize the recent historical series and its historical context texts into one forecast-relevant context state for forecast_time_range.

The context texts are noisy historical evidence, not a list of events that must be used. Understand them chronologically together with the target series. Distinguish concrete external drivers from target-value reports, generic background, local or mismatched evidence, repeated descriptions of the same event, and events whose effects are already visible and completed in the history. Produce one integrated assessment rather than selecting or separately reasoning over many texts.

Reason in this order:
1. Understand the exact target variable and the historical series: its level, trend, volatility, seasonal pattern, late turning points, and state at the end of input_time_range.
2. Relate the context texts to the observed historical evolution. Identify whether they explain an already-observed movement, indicate an active or delayed external influence, show weakening or recovery, contradict one another, or provide no target-aligned information. Separate an event's accumulated effect on the current level from evidence that it will cause further change after the forecast origin.
3. Decide whether any contextual influence remains relevant for forecast_time_range. Summarize its mechanism, current phase, likely next pressure on the target from the forecast origin, plausible timing, and scope. The next pressure may oppose the event's original effect during recovery. Do not infer a numeric offset and do not compare against a base forecast, which is handled in Stage2.

The dataset information is as follows.
[Dataset Description]
{dataset_description}
[Dataset Guidance]
{context_guidance}
Use Dataset Guidance to understand domain mechanisms, target scope, and target-variable semantics, not as a keyword list.

The forecasting setup is as follows.
[Forecasting Setup]
target_variable: {target_variable}
input_length: {input_length}
prediction_length: {prediction_length}
frequency: {frequency}
input_time_range: {input_time_range}
forecast_time_range: {forecast_time_range}
variable_description: {variable_description}

The historical time series is as follows.
[History Time Series]
reference_values at {time_0}: {initial_values_text}
calculation: each table value is computed as the current value minus the reference value at {time_0}.
{time_series_dfloader_text}

The reference values and table values are scaled values, not raw real-world measurements. Interpret them only for relative temporal changes.

The historical context texts are as follows.
[Context Texts]
{context_texts_dfloader_text}

The output must be exactly one valid JSON object following the structured template below. Do not include Markdown code fences or any extra text outside the JSON.
[Output]
{template}

[Context Status]
- `none`: the texts provide no credible target-aligned influence that remains relevant beyond the observed history.
- `weak`: a relevant influence is plausible, or an event still explains the current level, but its additional effect after the forecast origin is uncertain.
- `active`: concrete target-aligned evidence supports an active, delayed, emerging, or recovery-phase influence that can plausibly alter the target path after the forecast origin.

[Notes]
- context_summary must synthesize the net contextual state after considering all texts; do not enumerate every candidate event.
- Treat later historical values and later texts as evidence about whether an earlier event persisted, weakened, reversed, or ended.
- Give the latest observed turning point substantial weight. Earlier repeated reports that explain the pre-turn movement are not evidence that the same movement will resume unless later concrete evidence supports it.
- Do not invent conditional escalation, reversal, or persistence merely because it is possible. Describe the net pressure supported at the forecast origin.
- A target can remain unusually high or low while recovering in the opposite direction. State both the remaining level effect and the likely next movement when they differ.
- A turning point visible only in the numerical history helps identify the current phase but is not itself external contextual evidence. Do not mark the context `active` unless the texts add credible information about the path after that observed point.
- An ongoing condition that merely explains the current abnormal level is `weak` when the texts do not show that it will further change the target or alter the ongoing recovery after the forecast origin.
- Reports of historical target values usually describe what is already reflected in the series and are not by themselves forward evidence.
- Respect the target's geographic, population, variable, and temporal scope. Local or related-domain evidence is weak unless it plausibly affects the stated aggregate target.
- Use `none` when the only defensible conclusion is the endogenous historical pattern already visible in the series.
- evidence_times should contain only the timestamps essential to the synthesized conclusion and may be empty.
``
