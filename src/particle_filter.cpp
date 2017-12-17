/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 100;
    particles.resize(num_particles);
    weights.resize(num_particles);
    std::default_random_engine gen;
    std::normal_distribution<double> dist_x(x, std[0]);
    std::normal_distribution<double> dist_y(y, std[1]);
    std::normal_distribution<double> dist_theta(theta, std[2]);
    for (int i = 0; i < num_particles; i++) {
        Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1 / num_particles;
        particles[i] = p;
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/
    // Gaussian noise parameter for initialization
    std::default_random_engine gen;

    for (size_t i = 0; i < num_particles; ++i) {
        double x_old = particles[i].x;
        double y_old = particles[i].y;
        double theta_old = particles[i].theta;

        double theta_pred, x_pred, y_pred;

        if (abs(yaw_rate) > 1e-6) {
            theta_pred = theta_old + yaw_rate * delta_t;
            x_pred = x_old + velocity / yaw_rate * (sin(theta_pred) - sin(theta_old));
            y_pred = y_old + velocity / yaw_rate * (cos(theta_old) - cos(theta_pred));
        } else {
            theta_pred = theta_old;
            x_pred = x_old + velocity * delta_t * cos(theta_old);
            y_pred = y_old + velocity * delta_t * sin(theta_old);
        }
        normal_distribution<double> dist_x(x_pred, std_pos[0]);
        normal_distribution<double> dist_y(y_pred, std_pos[1]);
        normal_distribution<double> dist_theta(theta_pred, std_pos[2]);
        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.
    for (unsigned int i = 0; i < observations.size(); i++) {
        LandmarkObs o = observations[i];
        double min_dist = numeric_limits<double>::max();
        int map_id = -1;
        for (unsigned int j = 0; j < predicted.size(); j++) {
            LandmarkObs p = predicted[j];
            double cur_dist = dist(o.x, o.y, p.x, p.y);
            if (cur_dist < min_dist) {
                min_dist = cur_dist;
                map_id = p.id;
            }
        }
        observations[i].id = map_id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    for (int i = 0; i < num_particles; i++) {
        double p_x = particles[i].x;
        double p_y = particles[i].y;
        double p_theta = particles[i].theta;
        vector<LandmarkObs> predictions;
        for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
            // get id and x,y coordinates
            int lm_id = map_landmarks.landmark_list[j].id_i;
            float lm_x = map_landmarks.landmark_list[j].x_f;
            float lm_y = map_landmarks.landmark_list[j].y_f;
            if (dist(lm_x, lm_y, p_x, p_y) <= sensor_range) {
                predictions.push_back(LandmarkObs{lm_id, lm_x, lm_y});
            }
        }
        vector<LandmarkObs> transformed_obs;
        for (unsigned int j = 0; j < observations.size(); j++) {
            double t_x = cos(p_theta) * observations[j].x - sin(p_theta) * observations[j].y + p_x;
            double t_y = sin(p_theta) * observations[j].x + cos(p_theta) * observations[j].y + p_y;
            transformed_obs.push_back(LandmarkObs{observations[j].id, t_x, t_y});
        }
        dataAssociation(predictions, transformed_obs);
        particles[i].weight = 1.0;
        for (int j = 0; j < transformed_obs.size(); j++) {
            double pr_x, pr_y;
            LandmarkObs obs = transformed_obs.at(j);
            for (unsigned int k = 0; k < predictions.size(); k++) {
                if (predictions[k].id == obs.id) {
                    pr_x = predictions[k].x;
                    pr_y = predictions[k].y;
                }
            }
            double norm_factor = 2 * M_PI * std_landmark[0] * std_landmark[1];
            double prob = exp(
                    -((pow(obs.x - pr_x, 2) + pow(obs.y - pr_y, 2)) / (2 * std_landmark[0] * std_landmark[1])));
            particles[i].weight *= prob / norm_factor;
        }

    }

    double norm_factor = 0.0;
    for (const auto &particle : particles)
        norm_factor += particle.weight;

    for (int i = 0; i < num_particles; i++) {
        particles[i].weight /= (norm_factor + numeric_limits<double>::epsilon());
        weights[i] = particles[i].weight;
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    std::discrete_distribution<int> d(weights.begin(), weights.end());
    std::vector<Particle> weighted_sample(num_particles);
    std::default_random_engine gen;
    for (int i = 0; i < num_particles; ++i) {
        int j = d(gen);
        Particle sample_p;
        sample_p.x = particles.at(j).x;
        sample_p.y = particles.at(j).y;
        sample_p.theta = particles.at(j).theta;
        sample_p.weight = particles.at(j).weight;
        weighted_sample.at(i) = sample_p;
    }
    particles = weighted_sample;
}

Particle ParticleFilter::SetAssociations(Particle &particle, const std::vector<int> &associations,
                                         const std::vector<double> &sense_x, const std::vector<double> &sense_y) {
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best) {
    vector<double> v = best.sense_x;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best) {
    vector<double> v = best.sense_y;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}
