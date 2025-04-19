# file: .../scheduler.py
#
# This class enscapulsates scheduling functions in the IMLD app.
#------------------------------------------------------------------------------

# import required system modules
#
from flask_apscheduler import APScheduler

# import required blueprint modules
#
from .blueprint import clean_cache

# create APS Schedule instance
#
scheduler = APScheduler()

# add a scheduled job to clean the cache every 5 minutes
#
scheduler.add_job(id='clean_cache', func=clean_cache, trigger='interval', seconds=300)
