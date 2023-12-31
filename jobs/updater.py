from apscheduler.schedulers.background import BackgroundScheduler


def start():
    from .jobs import aggregate_and_retrain_model, categorize_uncategorized_images
    scheduler = BackgroundScheduler()
    scheduler.add_job(aggregate_and_retrain_model, 'interval', days=7)
    scheduler.add_job(categorize_uncategorized_images, 'interval', minutes=1)

    scheduler.start()
