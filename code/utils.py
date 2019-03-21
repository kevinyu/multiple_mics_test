def datetime2str(dt):
    s = (
        dt.hour * (1000 * 60 * 60) +
        dt.minute * (1000 * 60) +
        dt.second * 1000 +
        dt.microsecond / 1000
    )
    return "{}_{}".format(
        dt.strftime("%y%m%d"),
        int(s)
    )
