import datetime


def MakePastFuture(base, numdays_past, numdays_future):
    past = [base + datetime.timedelta(days=x) for x in range(numdays_past)]
    future = [past[-1] + datetime.timedelta(days=x) for x in range(1, numdays_future+1)]
    return past, future