import logging

def configure_sentry(settings, logger=None):
    if logger is None:
        logger = logging.getLogger("lisa.observability")

    dsn = getattr(settings, "sentry_dsn", None)
    if not dsn:
        logger.info("Sentry DSN not set; observability tracing disabled.")
        return

    try:
        import sentry_sdk
        
        sentry_sdk.init(
            dsn=dsn,
            environment=getattr(settings, "sentry_environment", "production"),
            release=getattr(settings, "sentry_release", None),
            traces_sample_rate=float(getattr(settings, "sentry_traces_sample_rate", 0.0)),
            profiles_sample_rate=float(getattr(settings, "sentry_profiles_sample_rate", 0.0)),
            send_default_pii=bool(getattr(settings, "sentry_send_default_pii", False)),
        )
        logger.info("Successfully initialized Sentry SDK.")
    except ImportError:
        logger.warning("sentry-sdk package not installed. Skipping sentry initialization.")
    except Exception as e:
        logger.error(f"Failed to initialize Sentry: {e}")
