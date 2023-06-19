CERTFILE=$1
KEYFILE=$2
source myenv/bin/activate
export FLASK_APP=app.py
gunicorn --certfile=$CERTFILE --keyfile=$KEYFILE --bind 0.0.0.0:5000 app:app


