#!/usr/bin/env python

import pinocchio as pin
from pinocchio.utils import *

import os
import tf
import rospy
import numpy as np

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState  

import matplotlib.pyplot as plt


def to_latex(num):
    """Convert a number to LaTeX scientific notation."""
    if 'e' in f'{num}':
        base, exponent = f'{num:.1e}'.split('e')
        exponent = int(exponent)  # Convert exponent to integer to remove leading zeros
        if base == '1.0':
            return rf'10^{{{exponent}}}'
        return rf'{base} \times 10^{{{exponent}}}'
    else:
        return f'{num}'

def plot(controller):
    desired_pos = np.array(controller.plt_data['desired_pos'])
    actual_pos = np.array(controller.plt_data['actual_pos'])
    est_pos = np.array(controller.plt_data['est_pos'])
    torques = np.array(controller.plt_data['torques'])
    slipping_est = np.array(controller.plt_data['est_slip'])

    # Create a large figure to hold all subplots
    fig = plt.figure(figsize=(10, 7))  # using constrained_layout for better spacing

    # Add a grid for subplots
    gs = fig.add_gridspec(3, 2)
    # Adjust displacement between subplots
    gs.update(hspace=0.6)

    # Position and Velocity Tracking
    ax1 = fig.add_subplot(gs[:2, 0])
    # Wheel Speeds
    ax6 = fig.add_subplot(gs[0, 1])
    # Wheel Torques
    ax7 = fig.add_subplot(gs[1, 1])
    # Error Analysis (combined)
    ax8 = fig.add_subplot(gs[2, 0])
    # Slipping Parameters Estimation
    ax9 = fig.add_subplot(gs[2, 1])

    # Add main title
    # fig.suptitle('Robot Position and Control Analysis', fontsize=20, fontweight='bold')

    # Add a subtitle below the main title
    param_info = \
        f'$\partial t={controller.dt}$' +\
        f', $\\alpha={controller.alpha}$' +\
        f', $R=I_d \cdot {to_latex(controller.R_scale)}$' +\
        f', $Q=I_d \cdot {to_latex(controller.Q_scale)}$' +\
        f', $\\mathbf{{k}} = [{controller.k1}, {controller.k2}, {controller.k3}, {controller.k4}, {controller.k5}]$' 
    fig.text(0.5, 0.90, f'{param_info}', fontsize=14, ha='center')

    # Adjust the top margin to make space for the main title and subtitle
    plt.subplots_adjust(top=0.85)

    # Plot position and velocity tracking
    ax1.plot(desired_pos[:, 0], desired_pos[:, 1], 'darkorange', label='$\\mathbf{p}_d$')
    ax1.plot(actual_pos[:, 0], actual_pos[:, 1], 'royalblue', linestyle='--', label='$\\mathbf{p}$')
    ax1.plot(est_pos[:, 0], est_pos[:, 1], 'crimson', linestyle=':', label='$\\mathbf{p}_e$')
    ax1.set_title('Position Tracking')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.legend()
    ax1.grid(True)

    # Plot wheel speeds
    ax6.plot(actual_pos[:, 3], 'darkorange', label='$\omega_{L}$')
    ax6.plot(actual_pos[:, 4], 'royalblue', label='$\omega_{R}$')
    ax6.set_title('Wheel Speeds')
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Speed')
    ax6.legend()
    ax6.grid(True)

    # Plot wheel torques
    ax7.plot(torques[:, 0], 'crimson', label='$\\tau_{L}$')
    ax7.plot(torques[:, 1], 'mediumseagreen', label='$\\tau_{R}$')
    ax7.set_title('Wheel Torques')
    ax7.set_xlabel('Time')
    ax7.set_ylabel('Torque')
    ax7.legend()
    ax7.grid(True)

    # Error Analysis (combining different metrics into a single subplot for clarity)
    da_error_x = desired_pos[:, 0] - actual_pos[:, 0]
    da_error_y = desired_pos[:, 1] - actual_pos[:, 1]
    da_error_norm = np.sqrt(da_error_x**2 + da_error_y**2)
    ae_error_x = actual_pos[:, 0] - est_pos[:, 0]
    ae_error_y = actual_pos[:, 1] - est_pos[:, 1]
    ae_error_norm = np.sqrt(ae_error_x**2 + ae_error_y**2)
    ax8.plot(da_error_norm, 'darkorange', label='$\\|| \\mathbf{e}_{d} \\||$')
    ax8.plot(ae_error_norm, 'royalblue', label='$\\|| \\mathbf{e}_{e} \\||$')
    ax8.set_title('Error Analysis')
    ax8.set_xlabel('Time')
    ax8.set_ylabel('Error Magnitude')
    ax8.legend()
    ax8.grid(True)

    # Plot slipping parameters
    ax9.plot(slipping_est[:, 0], 'royalblue', label='$i_{L_e}$')
    ax9.plot(slipping_est[:, 1], 'mediumseagreen', label='$i_{R_e}$')
    ax9.set_title('Slipping Parameters Estimation')
    ax9.set_xlabel('Time')
    ax9.set_ylabel('Slippage Value')
    ax9.legend()
    ax9.grid(True)

    # Adjust the layout to prevent overlap
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f'plots/dt_{controller.dt}_alpha_{controller.alpha}_R_{controller.R_scale}_Q_{controller.Q_scale}_k_{controller.k1}_{controller.k2}_{controller.k3}_{controller.k4}_{controller.k5}{"_bezier" if controller.use_bezier else ""}')
    # Save both as PNG and PDF
    plt.savefig(
        f'{filename}.png',
        bbox_inches='tight')
    plt.savefig(
        f'{filename}.pdf',
        bbox_inches='tight')
    
    plt.show()


class ControllerV2:
    def __init__(self) -> None:
        # Define state vector dimention and measurement dimension
        self.n = 7          # State dimension
        self.m = 5          # Measurement dimension

        # Robot specs
        self.b = 0.2022     # Wheel distance
        self.r = 0.0985     # Wheel radius

        # Set simulation duration
        self.duration = 280

        # Define target linear and angular reference velocities
        self.v_r = 0.8
        self.omega_r = 0.2
        # Set if the controller should use Bézier curve
        self.use_bezier = False
        # Initialize reference trajectory
        self.x_r = 0
        self.y_r = 0
        self.theta_r = 0
        # Initialize torques
        self.tau = np.array([0, 0])  
        # Define time step
        self.dt = 0.001
        
        # Initialize data for plotting
        self.plt_data = {
            'desired_pos': [],
            'actual_pos': [],
            'est_pos': [],
            'torques': [],
            'est_slip': []
        }

        # Load the XML file
        self.xml = rospy.get_param("/robot_description")
        # Initialize Pinocchio robot model
        self.model = pin.buildModelFromXML(self.xml, pin.JointModelFreeFlyer())        
        # Initialize Pinocchio data
        self.data = self.model.createData()
        # Initialize robot configuration and velocity
        self.q = pin.neutral(self.model)
        self.v = pin.utils.zero(self.model.nv)
        
        # Call Centroidal Composite Rigid Body Algorithm
        pin.ccrba(self.model, self.data, self.q, self.v)
        # Extract centroidal mass and centroidal inertia matrix
        m = sum([inertia.mass for inertia in self.model.inertias])
        I_zz = self.data.Ig.inertia[2, 2] 
        self.M = np.array([[m, 0, 0], [0, m, 0], [0, 0, I_zz]])
 
        # Define controller parameters
        self.k1 = 1
        self.k2 = 8
        self.k3 = 2
        self.k4 = 3
        self.k5 = 3

        # Initialize state vector
        self.x_est = np.array([1, 1, np.pi/4, 0, 0, 0, 0]) + 0.01 * np.array([1, 1, 1, 1, 1, 1, 1]) * np.random.normal(0, 1, 1)
        self.P = np.eye(self.n)
        self.measures = np.zeros(self.m)

        # UKF parameters
        self.alpha = 0.001
        self.beta = 2
        self.kappa = 0
        self.lambda_ = self.alpha**2 * self.n # (self.n + self.kappa) - self.n

        # Weights for sigma points
        self.Wm = np.full(2 * self.n + 1, 1 / (2 * (self.n + self.lambda_)))
        self.Wc = np.full(2 * self.n + 1, 1 / (2 * (self.n + self.lambda_)))
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.Wc[0] = self.Wm[0] + (1 - self.alpha**2 + self.beta)

        # Process and measurement noise covariances
        self.Q_scale = 1e-5
        self.R_scale = 1e-4
        self.Q = np.eye(self.n) * self.Q_scale
        self.R = np.eye(self.m) * self.R_scale

        # Initialize ROS node
        rospy.init_node('controller_node', anonymous=True)
        # Initialize subscribers and publishers
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback)
        rospy.Subscriber("/joint_states", JointState, self.joint_states_callback)
        self.cmd_vel_publisher = rospy.Publisher('/mobile_base_controller/cmd_vel', Twist, queue_size=1)


    ##### CONTROLLER

    @staticmethod
    def bezier(t, points):
        # Cubic Bézier curve calculation
        b0 = (1 - t)**3
        b1 = 3 * t * (1 - t)**2
        b2 = 3 * t**2 * (1 - t)
        b3 = t**3
        return b0 * points[0] + b1 * points[1] + b2 * points[2] + b3 * points[3]

    def command(self) -> None:
        # Start with the predict phase to estimate the current state
        self.predict()
        self.update()

        # Extract estimated variables
        x_e = self.x_est[0]
        y_e = self.x_est[1]
        theta_e = self.x_est[2]
        omega_Le = self.x_est[3]
        omega_Re = self.x_est[4]
        i_Le = self.x_est[5]
        i_Re = self.x_est[6]

        # REFERENCE TRAJECTORY: eq. 10
        # Use Euler method to integrate the trajectory derived from reference velocities
        # We update theta_r last for consistency with Euler method
        self.x_r += self.dt * self.v_r * np.cos(self.theta_r)
        self.y_r += self.dt * self.v_r * np.sin(self.theta_r)
        self.theta_r += self.dt * self.omega_r        
        
        # TRACKING ERROR: eq. 11
        # Compute rotation matrix
        R_e = np.array([
            [np.cos(theta_e), np.sin(theta_e), 0],
            [-np.sin(theta_e), np.cos(theta_e), 0],
            [0, 0, 1]])
        # Normilize theta error
        theta_err = self.theta_r - theta_e
        norm_theta_err = np.arctan2(np.sin(theta_err), np.cos(theta_err))
        # Compute trajectory error
        e_t = np.array([self.x_r - x_e, self.y_r - y_e, norm_theta_err])
        # Compute tracking error
        e = R_e @ e_t.T
        
        # AUXILIAR VELOCITY: eq. 12
        omega = self.omega_r + self.v_r / 2 * (self.k3 * (e[1] + self.k3 * e[2]) + (np.sin(e[2]) / self.k2))
        v = self.v_r * np.cos(e[2]) - self.k3 * e[2] * omega + self.k1 * e[0]  

        # DESIRED VELOCITY: eq. 13
        # Compute T matrix and its inverse
        T = self.r / (2 * self.b) * np.array([
            [self.b * (1 - i_Le), self.b * (1 - i_Re)],
            [-2 * (1 - i_Le), 2 * (1 - i_Re)]])
        T_inv = np.linalg.inv(T)
        # Compute desired velocities ξd = [omega_Ld, omega_Rd] = T @ [v, omega].T
        omega_Ld, omega_Rd = T_inv @ np.array([v, omega])

        # BACKSTEPPING: eq. 7
        # TODO: compute derivative of omega_Ld, omega_Rd (ξd_dot)
        omega_Ld_dot = 0
        omega_Rd_dot = 0
        # Compute auxiliary input
        u = np.array([omega_Ld_dot, omega_Rd_dot]) - np.array([self.k4 * (omega_Le - omega_Ld), self.k5 * (omega_Re - omega_Rd)]) 

        # CONTROL INPUT: eq 4
        # Compute S matrix
        S = (1 / (2 * self.b)) * np.array([
            [self.b * self.r * (1 - i_Le) * np.cos(theta_e), self.b * self.r * (1 - i_Re) * np.cos(theta_e)],
            [self.b * self.r * (1 - i_Le) * np.sin(theta_e), self.b * self.r * (1 - i_Re) * np.sin(theta_e)],
            [-2 * self.r * (1 - i_Le), 2 * self.r * (1 - i_Re)]])
        # Compute B matrix
        B = np.array([
            [np.cos(theta_e), np.cos(theta_e)],
            [np.sin(theta_e), np.sin(theta_e)],
            [-self.b / 2, self.b / 2]])
        # Call Centroidal Composite Rigid Body Algorithm
        pin.ccrba(self.model, self.data, self.q, self.v)
        # Update momentum inertia matrix
        self.M[2,2] = self.data.Ig.inertia[2, 2]
        # Compute B_bar and M_bar
        B_bar = S.T @ B
        M_bar = S.T @ self.M @ S
        # Compute tau
        self.tau = np.linalg.inv(B_bar) @ M_bar @ u.T

        self.plt_data['desired_pos'].append(np.array([self.x_r, self.y_r, self.theta_r]))
        self.plt_data['actual_pos'].append(np.array(self.measures))
        self.plt_data['est_pos'].append(np.array([x_e, y_e, theta_e]))
        print("torques: ", self.tau[0], self.tau[1])
        if np.abs(self.tau[0])< 15 and np.abs(self.tau[1]) < 15:
            self.plt_data['est_slip'].append(np.array([i_Le, i_Re]))
            self.plt_data['torques'].append(np.array(self.tau))


    ###### UKF

    def f(self, x: np.ndarray) -> np.ndarray:
        # Extract estimated variables
        theta_e = x[2]
        omega_Le = x[3]
        omega_Re = x[4]
        i_Le = x[5]
        i_Re = x[6]
        
        # Compute S matrix
        S = (1 / (2 * self.b)) * np.array([
            [self.b * self.r * (1 - i_Le) * np.cos(theta_e), self.b * self.r * (1 - i_Re) * np.cos(theta_e)],
            [self.b * self.r * (1 - i_Le) * np.sin(theta_e), self.b * self.r * (1 - i_Re) * np.sin(theta_e)],
            [-2 * self.r * (1 - i_Le), 2 * self.r * (1 - i_Re)]])
        # Compute B matrix
        B = np.array([
            [np.cos(theta_e), np.cos(theta_e)],
            [np.sin(theta_e), np.sin(theta_e)],
            [-self.b / 2, self.b / 2]])

        # Call Centroidal Composite Rigid Body Algorithm
        pin.ccrba(self.model, self.data, self.q, self.v)
        # Update momentum inertia matrix
        self.M[2,2] = self.data.Ig.inertia[2, 2]
        # Compute B_bar and M_bar
        B_bar = S.T @ B
        M_bar = S.T @ self.M @ S
        # Compute ξd_dot
        omega_Le_dot, omega_Re_dot = np.linalg.inv(M_bar) @ B_bar @ self.tau.T
        
        # Compute T matrix
        T = self.r / (2 * self.b) * np.array([
            [self.b * (1 - i_Le), self.b * (1 - i_Re)],
            [-2 * (1 - i_Le), 2 * (1 - i_Re)]])
        # Compute v and omega (eq. 8)
        v, omega = T @ np.array([omega_Le, omega_Re])

        # Euler method to integrate
        x[0] += self.dt * v * np.cos(theta_e)
        x[1] += self.dt * v * np.sin(theta_e)
        x[2] += self.dt * omega
        x[3] += self.dt * omega_Le_dot
        x[4] += self.dt * omega_Re_dot
        x[5] += np.random.normal(0, np.sqrt(self.Q[5][5]))
        x[5] = np.clip(x[5], -0.5, 0.5)
        x[6] += np.random.normal(0, np.sqrt(self.Q[6][6]))
        x[6] = np.clip(x[6], -0.5, 0.5)
        
        return x

    def compute_sigma_points(self) -> np.array:
        # Calculate square root of P matrix using Cholesky Decomposition
        sqrt_P = np.linalg.cholesky((self.lambda_ + self.n) * self.P)

        # Initialize sigma points
        sigma_points = np.zeros((2 * self.n + 1, self.n))
        sigma_points[0, :] = self.x_est
        
        # Calculate sigma points
        for k in range(self.n):
            spread_vector = sqrt_P[:, k]
            sigma_points[k + 1, :] = self.x_est + spread_vector
            sigma_points[self.n + k + 1, :] = self.x_est - spread_vector
        
        return sigma_points

    def predict(self) -> None:
        # Predict the state using the UKF equations
        sigma_points = self.compute_sigma_points()
        sigma_points = np.apply_along_axis(self.f, 1, sigma_points)
        self.sigma_points = sigma_points

        # Compute the predicted state mean
        self.x_pred = np.sum(self.Wm[:, np.newaxis] * sigma_points, axis=0)
        
        # Compute the predicted state covariance
        self.P_pred = self.Q.copy()
        for i in range(2 * self.n + 1):
            y = sigma_points[i] - self.x_pred
            self.P_pred += self.Wc[i] * np.outer(y, y)

    def update(self):        
        # Compute the predicted measurement mean
        Z = self.sigma_points[:, :self.m]
        # Compute z_pred
        z_pred = np.sum(self.Wm[:, np.newaxis] * Z, axis=0)
        z_robot = self.measures #just for consistency
        
        # Compute the predicted measurement covariance
        P_zz = self.R.copy()
        for i in range(2 * self.n + 1):
            y = Z[i] - z_pred
            P_zz += self.Wc[i] * np.outer(y, y)
        
        # Compute the cross-covariance matrix
        P_xz = np.zeros((self.n, self.m))
        for i in range(2 * self.n + 1):
            P_xz += self.Wc[i] * np.outer(self.sigma_points[i] - self.x_pred, Z[i] - z_pred)
        
        # Compute the Kalman gain
        K = P_xz @ np.linalg.inv(P_zz)
        
        # Update the state with the measurement
        # measurement = z_robot
        self.x_est = self.x_pred + K @ (z_robot - z_pred)
        self.P = self.P_pred - K @ P_zz @ K.T


    ##### CALLBACKS
    
    def model_states_callback(self, msg):
        # Check if the message contains the required data
        if len(msg.pose) > 1 and len(msg.twist) > 1:
            # Update position and orientation
            self.measures[0] = msg.pose[1].position.x
            self.measures[1] = msg.pose[1].position.y
            quaternion = (
                msg.pose[1].orientation.x,
                msg.pose[1].orientation.y,
                msg.pose[1].orientation.z,
                msg.pose[1].orientation.w,
            )
            _, _, self.measures[2] = tf.transformations.euler_from_quaternion(quaternion)

    def joint_states_callback(self, msg):
        # Unwrap the message
        for name, position, velocity in zip(msg.name, msg.position, msg.velocity):
            # For each wheel joint, update the position and velocity
            if name == 'wheel_right_joint' or name == 'wheel_left_joint':
                self.q[self.model.getJointId(name)-1] = position
                self.v[self.model.getJointId(name)-1] = velocity
                self.measures[3] = self.v[self.model.getJointId('wheel_left_joint')-1]
                self.measures[4] = self.v[self.model.getJointId('wheel_right_joint')-1]


    ##### MAIN LOOP
              
    def run_loop(self):
        # Set the rate of this loop
        rate = rospy.Rate(1/self.dt)
        # Set start time and duration
        start_time = rospy.get_time()

        # Initialize Bézier curve parameters
        i, total_time = 0, 5000

        # Run the UKF until the duration is reached
        while not rospy.is_shutdown():
            # Bézier curve
            if self.use_bezier:
                t = i / total_time if i <= total_time else 1
                i += 1
                self.v_r = self.bezier(t, [3e-3, 3e-2, 8e-2, 3e-1])
                self.omega_r = self.bezier(t, [1e-3, 1e-2, 5e-2, 1e-1])
			
            # UKF + CONTROLLER
            self.command()

            # Compute the control input
            omega_L = self.x_est[3] + self.tau[0]/30 
            omega_R = self.x_est[4] + self.tau[1]/30 
            # Compute T matrix     
            T = self.r / (2 * self.b) * np.array([
                [self.b, self.b ],
                [-2, 2]])
            # Compute v and omega (eq. 8)
            v, omega = T @ np.array([omega_L, omega_R])
            # Create Twist message
            cmd_vel_msg = Twist()
            cmd_vel_msg.linear.x = v 
            cmd_vel_msg.angular.z = omega        
            # Publish the velocity command
            self.cmd_vel_publisher.publish(cmd_vel_msg)

            # Check if the duration is reached
            if rospy.get_time() - start_time >= self.duration:
                plot(self)
                rospy.loginfo("Simulation duration reached. Shutting down.")
                rospy.signal_shutdown("Simulation duration reached")

            # Sleep for the remainder of the loop
            rate.sleep()


if __name__ == '__main__':
    try:
        complete_controller = ControllerV2()
        complete_controller.run_loop()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    
