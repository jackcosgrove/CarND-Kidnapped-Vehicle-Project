/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <algorithm>
#include <iostream>
#include <numeric>
#include <limits>
#include <math.h>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  
  if (!is_initialized) {
    
    num_particles = 100;
    
    double std_x, std_y, std_theta;
    
    std_x = std[0];
    std_y = std[1];
    std_theta = std[2];
    
    default_random_engine gen;

    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);
    
    for (int i=0; i<num_particles; ++i) {
      double sample_x, sample_y, sample_theta;
      
      sample_x = dist_x(gen);
      sample_y = dist_y(gen);
      sample_theta = dist_theta(gen);
      
      Particle p;
      p.id = i;
      p.x = sample_x;
      p.y = sample_y;
      p.theta = sample_theta;
      p.weight = 1.0;
      
      particles.push_back(p);
      
      weights.push_back(p.weight);
    }
    
    is_initialized = true;
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  
  double s_xy = velocity * delta_t;
  double s_yaw = yaw_rate * delta_t;
  
  double std_x, std_y, std_yaw;
  
  std_x = std_pos[0];
  std_y = std_pos[1];
  std_yaw = std_pos[2];
  
  normal_distribution<double> dist_x(0.0, std_x);
  normal_distribution<double> dist_y(0.0, std_y);
  normal_distribution<double> dist_yaw(0.0, std_yaw);
  
  default_random_engine gen;
  
  for (int i=0; i<num_particles; ++i) {
    Particle p = particles[i];
    double yaw = p.theta;
    
    if (fabs(yaw_rate) < 1e-5) {
      p.x += s_xy * cos(yaw);
      p.y += s_xy * sin(yaw);
    } else {
      double rate_ratio = velocity / yaw_rate;
      p.x += rate_ratio * (sin(yaw + s_yaw) - sin(yaw));
      p.y += rate_ratio * (cos(yaw) - cos(yaw + s_yaw));
      p.theta += s_yaw;
    }
    
    double sample_x, sample_y, sample_yaw;
    
    sample_x = dist_x(gen);
    sample_y = dist_y(gen);
    sample_yaw = dist_yaw(gen);
    
    p.x += sample_x;
    p.y += sample_y;
    p.theta += sample_yaw;
    
    particles[i] = p;
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

  for (int i=0; i<observations.size(); ++i) {
    
    double min_dist = numeric_limits<double>::max();
    
    for (int j=0; j<predicted.size(); ++j) {
      
      double distance_squared = pow(observations[i].x - predicted[j].x, 2) + pow(observations[i].y - predicted[j].y, 2);
      
      if (min_dist > distance_squared) {
        min_dist = distance_squared;
        
        observations[i].id = j;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
  
  double std_x, std_y;
  
  std_x = std_landmark[0];
  std_y = std_landmark[1];
  
  vector<LandmarkObs> landmarks;
  vector<LandmarkObs> landmarks_in_range;
  
  for (int i=0; i<map_landmarks.landmark_list.size(); ++i) {
    
    Map::single_landmark_s list_landmark = map_landmarks.landmark_list[i];
    
    LandmarkObs landmark;
    landmark.id = list_landmark.id_i;
    landmark.x = list_landmark.x_f;
    landmark.y = list_landmark.y_f;
    
    landmarks.push_back(landmark);
  }
  
  for (int i=0; i<num_particles; ++i) {
    
    Particle p = particles[i];
    
    landmarks_in_range.clear();
    for (int j=0; j<landmarks.size(); ++j) {
      
      double distance = dist(landmarks[j].x, landmarks[j].y, p.x, p.y);
      if (distance <= sensor_range) {
        landmarks_in_range.push_back(landmarks[j]);
      }
    }
    
    double cos_theta = cos(p.theta);
    double sin_theta = sin(p.theta);
    
    vector<LandmarkObs> transformed_observations;
    
    for (int j=0; j<observations.size(); ++j) {
      
      LandmarkObs observation = observations[j];
      
      LandmarkObs transformed_observation;
      transformed_observation.x = p.x + observation.x * cos_theta - observation.y * sin_theta;
      transformed_observation.y = p.y + observation.x * sin_theta + observation.y * cos_theta;
      
      transformed_observations.push_back(transformed_observation);
    }
    
    dataAssociation(landmarks_in_range, transformed_observations);
    
    double weight = 1.0;
    
    double weight_coefficient = 0.5 / M_PI / std_x / std_y;
    
    double x_denom = 2.0 * std_x * std_x;
    double y_denom = 2.0 * std_y * std_y;
    
    for (int j=0; j<transformed_observations.size(); ++j) {
      
      LandmarkObs obs = transformed_observations[j];
      LandmarkObs landmark = landmarks_in_range[obs.id];

      double exponent = exp(-(pow(obs.x-landmark.x, 2)/x_denom + pow(obs.y-landmark.y, 2)/y_denom));
      weight *= weight_coefficient * exponent;
    }
    
    particles[i].weight = weight;
    weights[i] = weight;
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  random_device rd;
  mt19937 generator(rd());
  discrete_distribution<> d(weights.begin(), weights.end());
  
  vector<Particle> new_particles;
  
  for (int i=0; i<num_particles; ++i) {
    
    new_particles.push_back(particles[d(generator)]);
  }
  
  particles = new_particles;
}

void ParticleFilter::write(string filename) {
	// You don't need to modify this file.
	ofstream dataFile;
	dataFile.open(filename, ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
