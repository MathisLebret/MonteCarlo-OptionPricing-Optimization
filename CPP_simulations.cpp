/** Build/Compilation commands:
 *
 * g++ -O3 -march=native -funroll-loops -ffast-math -flto -Wall -std=c++17 %f -o %e -fopenmp
 *
 * Explanation of flags used:
 * -O3             : Enable high-level optimizations for performance
 * -march=native   : Optimize code for the architecture of the compiling machine
 * -funroll-loops  : Unroll loops to improve runtime speed
 * -ffast-math     : Allow faster but less strict floating-point math optimizations
 * -flto           : Enable Link Time Optimization across compilation units
 * -Wall           : Enable all compiler warnings
 * -std=c++17      : Use the C++17 standard
 * -fopenmp        : Enable OpenMP for parallel programming support
 */

#include <iostream> // for basic c++
#include <vector> // for vectors use
#include <cmath> // for maths (exp, sqrt...)
#include <random> // for Normal law simulation
#include <algorithm> // for max formula
#include <numeric> // for sums
#include <chrono> // for time measurements
#include <omp.h> // for parallelisation with OpenMP
#include <Eigen/Dense> // for vectorized operations (https://eigen.tuxfamily.org/)
#include <EigenRand/EigenRand> // to draw a matrix/vector of random variables (https://bab2min.github.io/eigenrand/v0.5.1/en/index.html)
#include <functional> // to use functions as arguments

class Call{
	public:
		// Constructor (to initialize the call option data)
		Call(double S0, double K, double r, double sigma, double T, int steps)
		: normal_dist(0.0, 1.0), S0(S0), K(K), r(r), sigma(sigma), T(T), steps(steps)
		{
			generator.seed(std::random_device{}()); // setting the random number generator with a random seed
		}
		
		
		/** First version : Naive C++
		 *	This implementation is similar to Naive Python version (two nested loops), 
		 *	The use of 'auto&' in the first loop avoids copying elements,
		 *	which slightly improves runtime performance.
		*/
		
		double bs_simulation_naive(int N){

			std::vector<double> payoffs(N, 0.0);
			double dt = T/steps;
			const double drift = (r-sigma*sigma/2.0)*dt;
			const double diffusion = sigma * std::sqrt(dt);
			
			for (auto& payoff : payoffs){ // using auto& improves performance (saving payoffs by reference, not copying it)
				double St = S0;
				for (int i = 0; i < steps; ++i){
					St *= std::exp(drift + diffusion * normal_dist(generator));
				}
				payoff = std::max(St - K, 0.0);				
			}
			
			double C = std::exp(-r * T) * (1.0/N) * std::accumulate(payoffs.begin(), payoffs.end(), 0.0);
			
			return C;
		}
		
		/** Second version : OpenMP parallelization
		 * 	'#pragma omp parallel for' gives the compiler the instruction to use OpenMP to parallelize the for loop.
		 * 	OpenMP distributes iterations across threads (i.e, improves performance by using multiple CPU cores).
		 * 	Parallelization was made possible because paths are independent.
		*/
		
		double bs_simulation_openMP(int N){

			std::vector<double> payoffs(N, 0.0);
			double dt = T/steps;
			const double drift = (r-sigma*sigma/2)*dt;
			const double diffusion = sigma * std::sqrt(dt);
			
			#pragma omp parallel for
			for (int i = 0; i < N; ++i){ // iteration with i due to parallelisation
				std::default_random_engine local_gen(std::random_device{}());
				std::normal_distribution<double> local_normal(0.0, 1.0);
				
				double St = S0;
				for (int i = 0; i < steps; ++i){			
					St *= std::exp(drift + diffusion * normal_dist(local_gen));
				}
				payoffs[i] = std::max(St - K, 0.0);				
			}
			
			double C = std::exp(-r * T) * (1.0/N) * std::accumulate(payoffs.begin(), payoffs.end(), 0.0);
			
			return C;
		}
		
		
		/** Third version: Eigen Vectorization
		 * 	This version is closer to the NumPy vectorization.
		 * 	We use the Eigen library (with the Dense module).
		 * 	However, Eigen doesn't provide random simulations from the N(0,1) distribution.
		 * 	Hence, we have to use nested loops.
		 * 	The simulation is done step by step.
		 * 	At each iteration, we generate a vector of N random variables, and generate the next value of St.
		 * 	All paths are generated simultaneously.
		*/
		
		double bs_simulation_eigen(int N){
		
			double dt = T/steps;
			const double drift = (r-sigma*sigma/2)*dt;
			const double diffusion = sigma * std::sqrt(dt);
			
			// Vector of initial value S0 for each path
			Eigen::VectorXd St = Eigen::VectorXd::Constant(N, S0);

			for (int step = 0; step < steps; ++step) {
				// For each step, we need to draw a vector of N simulations from the N(0,1)
				Eigen::VectorXd Z(N);
				for (int i = 0; i < N; ++i) {
					Z[i] = normal_dist(generator);
				}
				// and we simulate the next step for each path
				St = St.array() * (drift + diffusion * Z.array()).exp();
			}
			
			// Vector of payoffs from the last value of each path
			Eigen::VectorXd payoffs = (St.array() - K).max(0.0);

			// Call price
			double C = std::exp(-r * T) * payoffs.mean();

			return C;
		}
		
		
		/** Fourth version : Eigen + OpenMP
		 * 	This version is a slight change of the 3rd version, but the paths generation is parallelized with OpenMP.
		 * 	We divide the N paths in different chunks, for parallelization purpose,
		 * 	which ensures the use of CPU multi-cores.
		 */
		
		double bs_simulation_eigen_openMP(int N) {
			double dt = T / steps;
			const double drift = (r - sigma * sigma / 2) * dt;
			const double diffusion = sigma * std::sqrt(dt);

			// Create the vector where the payoffs will be stored
			Eigen::VectorXd payoffs = Eigen::VectorXd::Zero(N);

			// We divide N in chunks and multiple threads execute the following code
			#pragma omp parallel
			{
				// For each thread we use a local generator (to avoid race conditions)
				std::default_random_engine local_gen(std::random_device{}());
				std::normal_distribution<double> local_normal(0.0, 1.0);

				// Distribute the simulation paths among threads
				#pragma omp for
				for (int i = 0; i < N; ++i) {
					double St = S0;
					// Simulate the asset price path step-by-step and store it in payoffs[path]
					for (int step = 0; step < steps; ++step) {
						double Z = local_normal(local_gen);
						St *= std::exp(drift + diffusion * Z);
					}
					payoffs[i] = std::max(St - K, 0.0);
				}
			}

			// Calculate the call price
			double C = std::exp(-r * T) * payoffs.mean();
			return C;
		}


		/** Fifth version : EigenRand
		 * 	This version is an improvement of the 3rd version.
		 * 	It requires the use of a library called EigenRand.
		 * 	EigenRand gives us the possibility to generate the (N x steps) matrix of random variables following the N(0,1) distribution.
		 *  It is built the same way as the vectorized NumPy model.
		 */
		
		double bs_simulation_eigen_rand(int N) {
			// Local generator to be the same type as EigenRand
			Eigen::Rand::Vmt19937_64 rng(std::random_device{}());

			// Generation of a (N x steps)-matrix Z
			Eigen::ArrayXXd Z = Eigen::Rand::normal<Eigen::ArrayXXd>(N, steps, rng);
			
			// Log-returns increments 
			double dt = T / steps;
			double drift = (r - 0.5 * sigma * sigma) * dt;
			double diffusion = sigma * std::sqrt(dt);
			Eigen::ArrayXXd log_returns = drift + diffusion * Z;
			
			// Cumulative sum of log-returns for each path
			Eigen::ArrayXXd log_paths = log_returns;
			for (int col = 1; col < steps; ++col) { // we work by steps, therefore we cannot parallelize this loop
				log_paths.col(col) += log_paths.col(col - 1);
			}
			
			// Final price (vector of ST)
			Eigen::ArrayXd ST = S0 * (log_paths.col(steps - 1)).exp();
			
			// Payoffs
			Eigen::ArrayXd payoffs = (ST - K).max(0.0);
			
			double C = std::exp(-r * T) * payoffs.mean();

			return C;
		}
		
		/** Sixth version : EigenRand + OpenMP
		 * 	Here, we kept the idea of the 4th version (Eigen + OpenMP) but with EigenRand.
		 * 	We are not using a matrix of random variables, but each path is computed faster thanks to the EigenRand generator.
		 */
		
		double bs_simulation_eigen_rand_openMP(int N) {
		double dt = T / steps;
		const double drift = (r - 0.5 * sigma * sigma) * dt;
		const double diffusion = sigma * std::sqrt(dt);

		Eigen::VectorXd payoffs = Eigen::VectorXd::Zero(N);

		#pragma omp parallel
		{
			Eigen::Rand::Vmt19937_64 rng(std::random_device{}() + omp_get_thread_num());

			#pragma omp for
			for (int i = 0; i < N; ++i) {
				// Generating a #steps-size vector of N(0,1) random draws
				Eigen::ArrayXd Z = Eigen::Rand::normal<Eigen::ArrayXd>(steps, 1, rng);

				Eigen::ArrayXd log_returns = drift + diffusion * Z;

				// Cumulative sum of log-returns for each path
				for (int col = 1; col < steps; ++col) {
					log_returns(col) += log_returns(col - 1);
				}

				// Calculating last price of each path (S_T) and the payoff
				double ST = S0 * std::exp(log_returns(steps - 1));
				payoffs[i] = std::max(ST - K, 0.0);
			}
		}

		double C = std::exp(-r * T) * payoffs.mean();
		
		return C;
	}



	private:
	
		// Generator
		std::default_random_engine generator; 
		std::normal_distribution<double> normal_dist;
		
		// Parameters for a European Call Option
		double S0, K, r, sigma, T;
		int steps;
};



void convergence(Call callParameters, int k, int maxpower);

void measure_time(Call callParameters, std::function<double(int)> func, int N);


int main(){
	Call callParameters(100.0, 105.0, 0.05, 0.2, 1.0, 100);
	
	// Study of the convergence of the estimator Cn
	
	convergence(callParameters, 10000, 5);
	convergence(callParameters, 1000, 7);
	
	// Evaluating runtime for every model from N = 10^5 to N = 10^7
	
	std::vector<std::pair<std::string, std::function<double(int)>>> functions = {
		{"bs_simulation_naive", [&](int n) { return callParameters.bs_simulation_naive(n); }},
		{"bs_simulation_openMP", [&](int n) { return callParameters.bs_simulation_openMP(n); }},
		{"bs_simulation_eigen", [&](int n) { return callParameters.bs_simulation_eigen(n); }},
		{"bs_simulation_eigen_openMP", [&](int n) { return callParameters.bs_simulation_eigen_openMP(n); }},
		{"bs_simulation_eigen_rand", [&](int n) { return callParameters.bs_simulation_eigen_rand(n); }},
		{"bs_simulation_eigen_rand_openMP", [&](int n) { return callParameters.bs_simulation_eigen_rand_openMP(n); }},

	};
	
	for (auto [name, function]: functions){
		std::cout << "--For the model : " << name << std::endl;
	
		for (int power = 5; power <= 7; ++power){
			std::cout << " --- For N = = 10^" << power << std::endl;
			measure_time(callParameters, function, std::pow(10,power));
			std::cout << std::endl;
		}
		
		std::cout << std::endl << std::endl << std::endl;
	}
	
	// For Eigen + OpenMP and EigenRand + OpenMP, runtime is also evaluated for N = 10^8
	
	measure_time(callParameters, [&](int n) { return callParameters.bs_simulation_eigen_openMP(n); }, std::pow(10, 8));
	
	std::cout << std::endl << std::endl << std::endl;
	
	measure_time(callParameters, [&](int n) { return callParameters.bs_simulation_eigen_rand_openMP(n); }, std::pow(10, 8));
	
	return 0;
}



	/** This function was created to measure the accuracy of the estimator Cn based on the value of N.
	 *	Only the mean and standard deviation are given, the confidence interval was then calculated in a spreadsheet.
	 *  This function is not directly linked to the call option so it's not part of the class.
	 */

void convergence(Call callParameters, int k, int maxpower){
	
	std::vector<double> mean_price;
	std::vector<double> std_price;
	
	for (int power = 3; power <= maxpower; ++power){
		int N = std::pow(10, power);
	
		std::vector<double> prices;
		prices.reserve(k);
		
		for (int j = 0; j < k; ++j){
			prices.push_back(callParameters.bs_simulation_eigen_rand_openMP(N));
		}
		
	double mean = std::accumulate(prices.begin(), prices.end(), 0.0) / prices.size();

    double standard_deviation = std::sqrt(std::accumulate(prices.begin(), prices.end(), 0.0, [&](double acc, double x) 
		{ double m = std::accumulate(prices.begin(), prices.end(), 0.0) / prices.size(); return acc + (x - m) * (x - m); }) / prices.size());

    std::cout << "N = 10^" << power << " : Mean = " << mean
		<< ", Std Dev = " << standard_deviation << std::endl;

	}
}

	/** This function was created to measure the accuracy of the runtime of 10 simulations
	 *	It also gives the average price of the 10 simulations as a check to verify model's correctness.
	 *  This function is not directly linked to the call option so it's not part of the class.
	 */

void measure_time(Call callParameters, std::function<double(int)> func, int N){
	
	// We don't need the price but it is used to guarantee that the simulations don't give false results
	std::vector<double> prices;
	
	auto start = std::chrono::high_resolution_clock::now();
	
	for (int i = 0; i < 10; ++i) {
        prices.push_back(func(N));
    }
	
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Average time for a simulation in seconds : " << elapsed_seconds.count()/10.0 << " ." << std::endl
		<< "With an average price : " << std::accumulate(prices.begin(), prices.end(), 0.0) / prices.size() << std::endl;
	
}
