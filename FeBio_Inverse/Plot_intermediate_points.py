import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.optimize import curve_fit


intermediate_file = "D:\\Gordon\\Automate FEB Runs\\2024_10_28\\2024_10_29_intermediate.csv"
header_pairs = [('inner_y', 'inner_z'), ('outer_y', 'outer_z'), ('innerShape_x', 'innerShape_y'), ('outerShape_x', 'outerShape_y')]

def plotIntermediatePoints(numrows, file):
        """ TODO: This function will use the intermediate file and the number of rows to plot the
        # the points that are in that row. The idea is that it will go through each header and collect
        # the x,y, or z and then use the points to plot what those points are representing. This should
        include some type of way to read the header files and the row that it is currently on and then
        take those numbers, put it in an array, and then plot using each element in that array. """
        df = pd.read_csv(file)
        df = df.head(numrows)

        for pair in header_pairs:
                x_header, y_header = pair
                x_coords = []
                y_coords = []

                print(f"\nProcessing pair: {x_header} and {y_header}")

                for col in df.columns:
                        if col.startswith(x_header):
                                x_coords.append(df[col].values)
                                #print(f"Found x column: {col}")
                        elif col.startswith(y_header):
                                y_coords.append(df[col].values)
                                #print(f"Found y column: {col}")


                x_cords_flat = [cord for sublist in x_coords for cord in sublist]
                y_cords_flat = [cord for sublist in y_coords for cord in sublist]
                print("X_VALS: ", x_cords_flat)
                print("Y_VALS: ", y_cords_flat)

                if len(x_cords_flat) != len(y_cords_flat):
                        raise ValueError("Mismatch in number of x and y coordinates.")

                coordinates = list(zip(x_cords_flat,y_cords_flat))

                #print(coordinates)

                x_vals, y_vals = zip(*coordinates)

                plt.figure()


                plt.scatter(x_vals, y_vals, label=f'{x_header} vs {y_header}', color='blue', marker='o')

                plt.xlabel(f'{x_header}')
                plt.ylabel(f'{y_header}')
                plt.title(f'Plot ({x_header}, {y_header}) Points First {numrows} Rows')
                plt.legend()


                plt.show()



#plotIntermediatePoints(20, intermediate_file)



def find_circle_center(points):
        """Finds the approximate center of a circle given a set of points.

        Args:

            points: A list of 2D points.

        Returns:
            A tuple (x, y) representing the center of the circle.
        """

        x = points[:, 0]

        y = points[:, 1]

        x_m, y_m = np.mean(x), np.mean(y)

        def calc_R(x, x_m, y_m):
                return np.sqrt((x - x_m) ** 2 + (y - y_m) ** 2)
        R, _ = curve_fit(calc_R, x, y, p0=[x_m, y_m])
        x_m, y_m = R
        return x_m, y_m


def create_spline(points, center):
        """Creates a spline that passes through the given points.
        Args:
            points: A list of 2D points.
            center: A tuple (x, y) representing the center of the circle.

        Returns:
            A NumPy array of points representing the spline.
        """

        # Sort points by angle relative to the center

        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])

        sorted_points = points[np.argsort(angles)]

        # Use a spline interpolation method (e.g., cubic spline)
        tck, u = splprep(sorted_points.T, s=0)

        unew = np.linspace(0, 1, num=100)

        xnew, ynew = splev(unew, tck)

        return np.column_stack((xnew, ynew))


def find_equal_spaced_points(spline_points, num_points=9):
        """Finds equally spaced points along a spline.
        Args:

            spline_points: A NumPy array of points representing the spline.

            num_points: The number of points to find.



        Returns:

            A NumPy array of equally spaced points.

        """

        tck, u = splprep(spline_points.T, s=0)

        u_equal = np.linspace(0, 1, num_points)

        x_equal, y_equal = splev(u_equal, tck)

        return np.column_stack((x_equal, y_equal))


# Example usage

points = np.array([[1, 2], [3, 4], [5, 3], [4, 1]])

center = find_circle_center(points)

spline_points = create_spline(points, center)

# Find 9 equally spaced points
equal_spaced_points = find_equal_spaced_points(spline_points, num_points=9)
# Visualize the results (e.g., using matplotlib)

plt.scatter(points[:, 0], points[:, 1])

plt.scatter(center[0], center[1], color='red')

plt.plot(spline_points[:, 0], spline_points[:, 1], color='green')

plt.scatter(equal_spaced_points[:, 0], equal_spaced_points[:, 1], color='blue')

plt.show()


