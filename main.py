from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from catboost import Pool

app = FastAPI()

# here will be the models which we use in our service
models = {}

column_names = [
                    'date_depart_year',
                    'date_depart_month',
                    'date_depart_week',
                    'date_depart_day',
                    'date_depart_hour',
                    'fr_id',
                    'route_type',
                    'is_load',
                    'rod',
                    'common_ch',
                    'vidsobst',
                    'distance',
                    'snd_org_id',
                    'rsv_org_id',
                    'snd_roadid',
                    'rsv_roadid',
                    'snd_dp_id',
                    'rsv_dp_id'
                ]


class RouteRequest(BaseModel):
    st_code_snd: str  # departure station
    st_code_rsv: str  # arrival station
    date_depart_year: int
    date_depart_month: int
    date_depart_week: int
    date_depart_day: int
    date_depart_hour: int
    fr_id: int  # float
    route_type: int  # float
    is_load: int
    rod: int
    common_ch: int  # float
    vidsobst: int  # float
    # distance: float        # will be found as a distance between departure station and arrival station
    snd_org_id: int
    rsv_org_id: int
    snd_roadid: int
    rsv_roadid: int
    snd_dp_id: int
    rsv_dp_id: int


class RouteApiResponse(BaseModel):
    op_status_code: int
    op_status_desc: str
    response: object


def request_is_valid(request: RouteRequest):
    # TODO: here will be the check for consistency of all the identifiers like stations, railroads and regions
    return True


def calc_distance(depart_station_id: str, arrive_station_id: str):
    return 300


def prepare_request_vector(request: RouteRequest, distance: float):

    incoming_params = [
                        request.date_depart_year,
                        request.date_depart_month,
                        request.date_depart_week,
                        request.date_depart_day,
                        request.date_depart_hour,
                        request.fr_id,
                        request.route_type,
                        request.is_load,
                        request.rod,
                        request.common_ch,
                        request.vidsobst,
                        distance,
                        request.snd_org_id,
                        request.rsv_org_id,
                        request.snd_roadid,
                        request.rsv_roadid,
                        request.snd_dp_id,
                        request.rsv_dp_id
                      ]
    return pd.DataFrame(incoming_params, column_names)


def calc_travel_time(input_df: pd.DataFrame):
    pool_test = Pool(input_df)

    calc_model = models.get("time_predict")

    travel_time = 0.0

    if calc_model is not None:
        travel_time = calc_model.predict(pool_test)

    return travel_time


@app.on_event("startup")
async def startup_event():
    with open("model.pkl", "rb") as f:
        models["time_predict"] = pickle.load(f)


@app.get("/api/hello")
async def root():
    return {"message": "Hello World"}


@app.post("/api/route-time", response_model=RouteApiResponse)
async def calc_route_time(request: RouteRequest):
    status_code = 0
    status_desc = "Ok"
    print("received request")
    response = {}

    if request_is_valid(request):
        print("request is valid")
        distance = calc_distance(request.st_code_snd, request.st_code_rsv)
        input_df = prepare_request_vector(request, distance)
        input_df = input_df.transpose()
        travel_time_array = calc_travel_time(input_df)
        response["travel_time"] = travel_time_array[0]
    else:
        print("request validation failed")
        response["error"] = "Invalid input parameters"

    return {
        "op_status_code": status_code,
        "op_status_desc": status_desc,
        "response": response
    }
