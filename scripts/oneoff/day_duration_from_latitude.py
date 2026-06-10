import numpy as np
import matplotlib.pyplot as plt

def calculate_daylight_duration(latitude, day_of_year):
    """
    Calculates the duration of daylight for a given latitude and day of the year.

    Args:
        latitude (float): Latitude in degrees.
        day_of_year (int): Day of the year (1 to 365).

    Returns:
        float: Duration of daylight in hours.
    """
    
    # Convert latitude to radians
    lat_rad = np.radians(latitude)

    # Calculate solar declination angle in radians
    # The 284 is an offset to account for the solstices and equinoxes
    solar_declination_rad = np.radians(23.45 * np.sin(np.radians(360 / 365 * (day_of_year + 284))))

    # Calculate the hour angle (omega)
    # This formula can result in values slightly greater than 1 or less than -1
    # due to floating point inaccuracies, which causes issues with arccos.
    # We clip the value to ensure it's within the valid domain [-1, 1].
    
    arg_arccos = -np.tan(lat_rad) * np.tan(solar_declination_rad)
    arg_arccos = np.clip(arg_arccos, -1.0, 1.0)
    
    hour_angle = np.degrees(np.arccos(arg_arccos))

    # Duration of daylight in hours
    duration = 2/15 * hour_angle
    
    return duration

if __name__ == "__main__":
	latitudes = 48, 45, 35
	days = np.arange(1, 366)

	# Calculate daylight duration for each latitude
	durations = np.array([
		[calculate_daylight_duration(latitude, day) for day in days]
		for latitude in latitudes
	]).T

	# Plotting
	plt.figure(figsize=(10, 6))
	plt.plot(days, durations, label = [f"{latitude}°N" for latitude in latitudes])

	plt.title('Day duration depending on the latitude')
	plt.xlabel('Day of year')
	plt.ylabel('Day duration [h]')
	plt.grid(True)
	plt.legend()
	plt.xticks(np.arange(0, 366, 30)) # Show x-axis ticks every 30 days
	plt.xlim(0, 365)
	plt.ylim(0, 24)
	plt.tight_layout()
	plt.show()
