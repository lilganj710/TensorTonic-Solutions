from dataclasses import dataclass


@dataclass
class Config:
    '''Determines if/when retraining is triggered'''
    budget: int
    '''Budget is mutable, may change by retrain_cost upon training'''
    cooldown: int
    retrain_cost: int
    max_staleness: int
    '''Number of days before a forced retrain is triggered'''
    drift_threshold: float
    '''Keeping track of both drift (from some baseline) and performance
    Drift and performance are definitely correlated, but I can see how a model
    can drift from a baseline while still keeping (somewhat) good performance
    Still though, drift isn't intended, and warrants a retrain'''
    performance_threshold: float


@dataclass
class DailyStats:
    '''Stats of the model for each day'''
    day: int
    drift_score: float
    '''Again, a model may drift from some baseline while keeping decent performance.
    It's worthwhile to keep track of both'''
    performance: float


def retraining_policy(
        daily_stats_raw: list[dict[str, float]],
        config_raw: list[dict[str, float]]) -> list[int]:
    """
    Decide which days to trigger model retraining. Return the days on which the model is retrained
        Note that the "cooldown is initially satisfied"
        What about initial staleness? Can I just act like the model was first trained
        at time 0 here?
    """
    cur_config = Config(**config_raw)
    daily_stats = sorted(
        [DailyStats(**ds) for ds in daily_stats_raw],
        key=lambda daily_stats: daily_stats.day
    )
    retraining_days: list[int] = []
    last_retrain = 0
    initial_cooldown = True
    for day_stats in daily_stats:
        days_since_retrain = day_stats.day - last_retrain
        retraining_trigger = (
            day_stats.drift_score > cur_config.drift_threshold
            or day_stats.performance < cur_config.performance_threshold
            or days_since_retrain >= cur_config.max_staleness
        )
        if not retraining_trigger:
            continue
        retraining_happens = (
            (days_since_retrain >= cur_config.cooldown or initial_cooldown)
            and cur_config.budget >= cur_config.retrain_cost
        )
        if retraining_happens:
            retraining_days.append(day_stats.day)
            last_retrain = day_stats.day
            cur_config.budget -= cur_config.retrain_cost
            initial_cooldown = False
    return retraining_days