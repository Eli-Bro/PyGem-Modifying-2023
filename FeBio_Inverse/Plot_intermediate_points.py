import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import splprep, splev, CubicSpline
from scipy import interpolate
from scipy.optimize import curve_fit
import ShapeAnalysisVerification as sav
import math


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

        x = np.array([p[0] for p in points])

        y = np.array([p[1] for p in points])

        x_m, y_m = np.mean(x), np.mean(y)

        return x_m, y_m


'''def create_spline(points, center):
        """Creates a spline that passes through the given points.
        Args:
            points: A list of 2D points.
            center: A tuple (x, y) representing the center of the circle.

        Returns:
            A NumPy array of points representing the spline.
        """

        points = np.array(points)

        # Sort points by angle relative to the center
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])

        sorted_points = np.argsort(angles)
        sorted_angles = angles[sorted_points]
        sorted_xs = points[sorted_points, 0]
        sorted_ys = points[sorted_points, 1]

        curve_x = interpolate.UnivariateSpline(sorted_angles, sorted_xs, k=5)
        curve_y = interpolate.UnivariateSpline(sorted_angles, sorted_ys, k=5)

        spaced_angles = np.linspace(sorted_angles[0], sorted_angles[-1], num =100)

        xnew = curve_x(spaced_angles)
        ynew = curve_y(spaced_angles)


        # Use a spline interpolation method (e.g., cubic spline)
        #tck, u = splprep(sorted_points.T, s=0)

        #unew = np.linspace(0, 1, num=100)

        #xnew, ynew = splev(unew, tck)

        return np.column_stack((xnew, ynew))
'''
def create_spline(points, center):
        """Creates a spline that passes through the given points.
        Args:
            points: A list of 2D points.
            center: A tuple (x, y) representing the center of the circle.

        Returns:
            A NumPy array of points representing the spline.
        """
        points = np.array(points)

        # Sort points by angle relative to the center
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])

        # Sort the points to match them to the angle
        sorted_points = np.argsort(angles)
        sorted_angles = angles[sorted_points]
        sorted_xs = points[sorted_points, 0]
        sorted_ys = points[sorted_points, 1]

        # adds the starting point to the end to close the circle
        sorted_angles = np.append(sorted_angles, sorted_angles[0] + 2 * np.pi)
        sorted_xs = np.append(sorted_xs, sorted_xs[0])
        sorted_ys = np.append(sorted_ys, sorted_ys[0])


        '''curve_x = interpolate.UnivariateSpline(sorted_angles, sorted_xs, k=5)
        curve_y = interpolate.UnivariateSpline(sorted_angles, sorted_ys, k=5)'''
        # creates the splines for the xs and ys
        curve_x = CubicSpline(sorted_angles, sorted_xs, bc_type = 'periodic')
        curve_y = CubicSpline(sorted_angles, sorted_ys, bc_type = 'periodic')

        # defines the evenly spaced angles to for the spline points
        spaced_angles = np.linspace(0,2 * math.pi, num=10)
        spaced_angles = spaced_angles[:-1]
        print("Spaced angles: ", spaced_angles)

        #generates the x and y values using the splines
        xnew = curve_x(spaced_angles)
        ynew = curve_y(spaced_angles)
        print("xnew: ", xnew)
        print("ynew: ", ynew)
        # returns the two arrays as a 2D array
        return np.column_stack((xnew, ynew))

'''def find_equal_spaced_points(spline_points, center_inner,num_points=9):
        """Finds equally spaced points along a spline.
        Args:

            spline_points: A NumPy array of points representing the spline.

            num_points: The number of points to find.



        Returns:

            A NumPy array of equally spaced points.

        """

        tck, u = splprep(spline_points.T, s=0)

        u_equal = np.linspace(0, 1, num_points)
        print("u_equal: ",u_equal)

        x_equal, y_equal = splev(u_equal, tck)

        return np.column_stack((x_equal, y_equal))
'''

def find_equal_spaced_points(spline_points, center,num_points=9):
        """Finds points along the spline at set angles (0, 40, ..., 320 degrees).

        Args:
                spline_points: A NumPy array of points representing the spline.
                center: A tuple (x, y) representing the center of the circle.
                num_points: The number of points to find (default is 9, spaced by 40 degrees).

        Returns:
                A NumPy array of points at specified angles along the spline.
        """
         # Convert points to an array if not already
        points = np.array(spline_points)
        print("points: ", points)

        # Calculate angles of each point relative to the center
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        print("angles relative to center: ", angles)

        # Normalize angles to range [0, 2π)
        angles = (angles + 2 * np.pi) % (2 * np.pi)
        print("normalized angles: ", angles)
        # Define target angles from 0 to 320 degrees in radians
        target_angles = np.linspace(0, 2 * np.pi, num=num_points, endpoint=False)
        print("target_angles: ", target_angles)

        # Find closest spline point for each target angle
        selected_points = []
        for target_angle in target_angles:
                # Find index of the closest angle to the target angle
                closest_index = np.argmin(np.abs(angles - target_angle))
                selected_points.append(points[closest_index])

        # Return as a NumPy array
        print("selected points: ", selected_points)

        return np.array(selected_points)


def angle_spline_driver(inner_radius, outer_radius):
        inner_radius = sav.get_2d_coords_from_dictionary(inner_radius)
        outer_radius = sav.get_2d_coords_from_dictionary(outer_radius)

        print("here are the coords that are going to be going into function: ")
        print("inner radius: ", inner_radius)
        print("outer radius: ", outer_radius)

        # calculates the center of points
        center_inner = find_circle_center(inner_radius)
        center_outer = find_circle_center(outer_radius)

        #Gets the spline of those points
        spline_points_inner = create_spline(inner_radius, center_inner)
        spline_points_outer = create_spline(outer_radius, center_outer)

        # Finds the equally spaced points based off of the spline
        #equal_spaced_points_inner = find_equal_spaced_points(spline_points_inner, center_inner, 9)
        #equal_spaced_points_outer = find_equal_spaced_points(spline_points_outer, center_outer, 9)

        #converts to a np array if not already
        outer_radius = np.array(outer_radius)
        inner_radius = np.array(inner_radius)

        #plots the spline and any points that are needed on the graph.
        #plot_spline(center_inner, spline_points_inner, inner_radius)
        #plot_spline(center_outer, spline_points_outer, outer_radius)

        return spline_points_inner, spline_points_outer

def plot_spline(center, spline_points, radius):
        plt.scatter(radius[:, 0], radius[:, 1])

        plt.scatter(center[0], center[1], color='red')

        plt.plot(spline_points[:, 0], spline_points[:, 1], color='green')

        #plt.scatter(equally_spaced_points[:, 0], equally_spaced_points[:, 1], color='blue')

        plt.show()


# Example usage
"""
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
"""
