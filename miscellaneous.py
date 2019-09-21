from datetime import date

def get_today_string():
    today = date.today()
    month = today.month
    if month < 10:
        month = f"0{month}"
    day = today.day
    if day < 10:
        day = f"0{day}"
    key = f"{today.year}-{month}-{day}"
    return key
