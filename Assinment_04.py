def next_birthday(date, birthdays):
    '''
    Find the next birthday after the given date.
    '''
    month, day = date

    # Sort all birthday dates (month, day)
    sorted_dates = sorted(birthdays.keys())

    # First, look for a birthday later in the same year
    for bday in sorted_dates:
        if bday > (month, day):
            return bday, birthdays[bday]

    # If none found, wrap around to the first birthday next year
    first_birthday = sorted_dates[0]
    return first_birthday, birthdays[first_birthday]
