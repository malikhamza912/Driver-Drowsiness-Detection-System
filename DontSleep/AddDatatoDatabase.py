import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://smartdds-1fda9-default-rtdb.firebaseio.com/ "

})

ref = db.reference('driver')

data = {
    "321654":
        {
            "name": "MOIN SATTI",
            "bus_no" :"574",
            "starting_year": 2017,
            "total_attendance": 0,
            "last_attendance_time": "2022-12-11 00:54:34"


        }

}

for key, value in data.items():
    ref.child(key).set(value)