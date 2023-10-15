from apscheduler.schedulers.background import BackgroundScheduler


def start():
    from .jobs import aggregate_and_retrain_model, categorize_uncategorized_images
    scheduler = BackgroundScheduler()
    scheduler.add_job(aggregate_and_retrain_model, 'interval', minutes=5)  #TODO: review
    scheduler.add_job(categorize_uncategorized_images, 'interval', minutes=5)

    scheduler.start()
