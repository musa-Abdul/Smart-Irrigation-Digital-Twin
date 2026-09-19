# Smart Irrigation Digital Twin

This project investigates how a digital twin can combine soil-moisture measurements, environmental data, physical reasoning and machine learning to support irrigation monitoring and decision-making.

Soil moisture is influenced by several interacting processes. Rainfall and irrigation add water to the soil, while evaporation, plant water use and drainage remove it. In practice, monitoring is also affected by noisy measurements, missing observations and limited sensor coverage.

I built this project to explore whether physical information can be incorporated into a time-series forecasting system and whether the resulting forecasts can support more informed irrigation decisions under uncertainty.

## Research Question

Can soil-water information and machine learning be combined to produce useful short-term soil-moisture forecasts, and can those forecasts support irrigation decisions when sensor measurements and future conditions are uncertain?

## Dataset

The experiments use the Gallipoli precision-irrigation dataset:

**Time-Series Analysis and Learning Approaches for Soil Moisture Prediction in Precision Irrigation: Dataset and Code**

The field dataset contains soil-water-content measurements from six sensor ports together with rainfall, irrigation and temperature observations.

The six soil sensors represent measurements at two depths. Odd-numbered ports correspond to measurements at 30 cm, while even-numbered ports correspond to 60 cm.

For the integrated analysis, the available observations span:

**1 August 2022 to 28 February 2023**

After removing four exact duplicate timestamps, 11,766 observations remained.

The original irrigation series also contains missing observations. I therefore keep track of irrigation-data availability rather than assuming that every missing measurement represents zero irrigation.

Dataset DOI: **10.5281/zenodo.20665025**

## Digital Twin Framework

The prototype follows the flow:

**Field measurements → Data processing → Physical information → Time-series forecasting → Uncertainty estimation → Irrigation decision support**

The digital twin uses soil-moisture measurements together with rainfall, irrigation and temperature information to estimate how soil moisture may change over the following hour.

A simple soil-water balance provides the physical motivation:

**ΔS = P + I − ET − D**

where:

- **ΔS** represents the change in stored soil water
- **P** represents rainfall
- **I** represents irrigation
- **ET** represents evapotranspiration
- **D** represents drainage and other losses

The available dataset does not provide all variables required to identify each physical term directly. For that reason, the current implementation uses the water-balance relationship to guide feature construction and model interpretation rather than claiming to solve a complete mechanistic soil-water model.

## Data Exploration

The first part of the project examined sampling intervals, missing values, rainfall and irrigation events, sensor behaviour and temporal continuity.

Most measurements were recorded at approximately 15-minute intervals, although gaps occur in the series. Forecast sequences are therefore constructed only from continuous 15-minute observations rather than joining measurements across long gaps.

Exploratory analysis also showed that soil moisture is highly persistent during many periods, while relatively sharp wetting and drying events occur less frequently. This makes persistence an important forecasting baseline.

## Physics-Based Exploration

I first investigated whether short-term soil-moisture changes could be explained directly from rainfall, irrigation and temperature.

A simple linear relationship using these instantaneous variables provided little explanatory power. This suggested that soil-moisture response depends on temporal history, delayed water movement and other unobserved processes.

I therefore introduced previous-hour rainfall and irrigation information as physically motivated temporal features for the forecasting model.

These variables are treated as empirical indicators of recent water input. They are not interpreted as a dimensionally complete field-scale water balance because the units and spatial characteristics required for that interpretation are not fully available.

## Soil-Moisture Forecasting

The forecasting task predicts the change in Port1 soil moisture one hour into the future.

The model uses a three-hour history consisting of 12 consecutive 15-minute observations.

Inputs include:

- six soil-moisture sensor measurements
- rainfall
- recorded irrigation
- irrigation missingness information
- temperature
- previous-hour rainfall
- previous-hour irrigation
- previous-hour combined water-input information

The data are divided chronologically into training, validation and test periods. Feature and target scaling are fitted using the training data only.

## Physics-Guided LSTM

A recurrent neural network based on an LSTM is used for one-hour soil-moisture forecasting.

The current model is best described as a **physics-guided LSTM** rather than a full physics-informed neural network. Physical reasoning is introduced through water-balance-inspired temporal features, while the network itself is trained using the forecasting error.

The model predicts the one-hour change in soil moisture:

**future soil moisture = current soil moisture + predicted change**

This formulation allows the model to focus on the change in the system state rather than directly reproducing the absolute moisture level.

## Forecasting Results

Persistence proved to be a difficult baseline to beat because much of the dataset contains only small short-term changes in soil moisture.

In the original forecasting experiment, the physics-guided LSTM did not outperform persistence across the complete test period. However, examining meaningful soil-moisture changes separately revealed a more informative result.

For drying events with an absolute one-hour change of at least 0.001, the physics-guided model reduced mean absolute error by approximately **45.8%** relative to persistence.

The model performed less effectively during sudden wetting events and tended to underestimate some rapid moisture increases.

This result is important because it shows that aggregate forecasting metrics alone can hide differences between stable, drying and wetting conditions.

## Sensor Robustness

Real monitoring systems do not always provide complete or perfectly clean measurements. I therefore tested the forecasting system under several synthetic sensor disturbances.

The experiments included:

- randomly missing Port1 measurements
- continuous Port1 outages
- Gaussian noise applied to the soil sensors
- complete replacement of selected sensor inputs
- event-specific analysis during meaningful soil-moisture changes

Small random disturbances produced only minor changes in overall forecasting error.

More severe sensor removal produced non-monotonic results because the dataset contains many periods with very small soil-moisture changes. This reinforced the need to evaluate model behaviour during meaningful events rather than relying only on aggregate error metrics.

## Forecast Uncertainty

A point prediction alone does not show how reliable a forecast may be.

I therefore used split-conformal calibration based on validation residuals to construct a nominal 90% prediction interval.

The calibrated uncertainty margin was:

**0.00088183**

On the unseen test period:

- test forecasts: **1,948**
- observed interval coverage: **93.84%**
- forecasts outside the interval: **120**
- mean interval width: **0.00176366**

The observed coverage is higher than the nominal 90% level, indicating that the interval was conservative on this test period.

This value represents prediction-interval coverage, not forecasting accuracy.

## Irrigation Decision Support

The final part of the prototype investigates how soil-moisture information could support irrigation decisions.

Because the dataset does not provide enough information to convert each recorded irrigation action into a reliable field-scale water depth, irrigation is treated as an effective decision action rather than a field-ready prescription.

Two experimental operating references are derived from the training data:

- **Operating target: 0.249**
- **Lower reference: 0.180**

These values are experimental training-data references. They are not crop-specific field-capacity or wilting-point thresholds.

The decision experiment compares fixed scheduling, current-state decisions, forecast-based decisions and uncertainty-aware decisions.

A fixed schedule derived from the training irrigation frequency produced **200 decision actions** during the test decision period. Only 12 coincided with the 115 future low-moisture cases defined in the experiment.

The current-state and one-hour forecast strategies each used **119 actions** and detected all 115 defined low-moisture cases. The uncertainty-aware strategy used **120 actions** and also detected all 115 cases.

This corresponds to approximately a **40% reduction in decision actions** relative to the fixed-schedule reference.

The forecast-based strategy did not outperform the current-state strategy at the selected operating target. Over a one-hour horizon, the current measurement and forecast often resulted in the same decision.

These action reductions should not be interpreted as measured field water savings. The experiment demonstrates a decision framework and resource-risk trade-off rather than a validated field irrigation prescription.

## Interactive Dashboard

A Streamlit dashboard connects the main components of the prototype.

The dashboard provides:

- system and dataset overview
- six-sensor soil-moisture monitoring
- recent rainfall, irrigation and temperature information
- one-hour Port1 soil-moisture forecasting
- 90% prediction uncertainty
- experimental operating references
- uncertainty-aware irrigation decision support

The dashboard loads the saved forecasting model and preprocessing scalers rather than retraining the model each time it starts.

Run the dashboard with:

```bash
streamlit run dashboard/irrigation.py