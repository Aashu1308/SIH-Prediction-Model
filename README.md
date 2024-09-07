# Code for Shipping Price Prediction ML Model used for SIH Hackathon

## Authors: Aashutosh and Nawaz

## Data used is not real and only for simulation

Parameters used for estimation: Distance, Weight and Loading Meters

Flask App present for API calls. To be used at port 8080. Deployed as WSGI server using gunicorn.

Run the command following command to run using pre-defined configurations:
`gunicorn --config gunicorn_config.py app:app`

This sets thread count to 2, worker count to 2 and listens to all incoming traffic from port 8080

To use default configurations use:
`gunicorn app:app`

This listens to all incoming traffic from port 8000

POST request should contain: Weight, Origin City and Destination City
