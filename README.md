# Smart Irrigation Digital Twin

I started this project to explore how a digital twin could be used to monitor soil moisture and support irrigation decisions.

Soil moisture does not depend on irrigation alone. Rainfall, evaporation, drainage and changing weather conditions all affect the amount of water available in the soil. At the same time, field sensors are not always perfect. Measurements can be noisy, missing or taken at only a few locations.

My aim is to build a small irrigation digital twin that combines a physical soil-water model with machine learning. I want to see whether adding physical information to the learning model makes soil-moisture prediction more reliable, especially when the sensor data are incomplete or uncertain.

## Main question

Can a model that combines soil-water physics with machine learning predict soil moisture more reliably than a purely data-driven model, and can those predictions be useful for irrigation scheduling?

## Where I am now

I am starting with the data. Before building the prediction models, I want to understand the soil-moisture measurements, rainfall and irrigation events, sampling intervals, missing values and the general behaviour of the field data.

## What I plan to build

The project will be developed in stages:

- explore and clean the field data
- build a simple soil-water balance model
- establish basic forecasting baselines
- develop a time-series machine-learning model
- introduce physical constraints into the learning model
- test the models when sensor readings are noisy or missing
- use the forecasts to investigate irrigation scheduling and water use

The scope may change as I learn more from the data and the first modelling results.