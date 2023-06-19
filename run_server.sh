source myenv/bin/activate
export FLASK_APP=app.py
#python3 -m flask run --host=0.0.0.0
gunicorn --certfile=$CERTFILE --keyfile=$KEYFILE --bind 0.0.0.0:5000 app:app


