#include <julia.h>
#include <iostream>
#include <vector>
#include <sstream>
#include <chrono>

JULIA_DEFINE_FAST_TLS

class OptimizedJuliaEGGO {
private:
    bool models_loaded = false;
    
public:
    OptimizedJuliaEGGO() {
        jl_init();
        
        // Load the EGGO module
        jl_eval_string("push!(LOAD_PATH, \"/Users/mcclenaghan/.julia/dev/EGGO/src\")");
        jl_eval_string("import EGGO");
        
        if (jl_exception_occurred()) {
            jl_call2(jl_get_function(jl_base_module, "showerror"), 
                     jl_stderr_obj(), jl_exception_occurred());
            throw std::runtime_error("Failed to load EGGO");
        }
    }
    
    ~OptimizedJuliaEGGO() {
        jl_atexit_hook(0);
    }
    
    void load_models_once() {
        if (models_loaded) {
            std::cout << "Models already loaded, skipping..." << std::endl;
            return;
        }
        
        std::cout << "Loading EGGO models (one-time initialization)..." << std::endl;
        
        // Load models once and store them as global Julia variables
        jl_eval_string(R"(
            # Load models once and store globally
            const MODEL_NAME = :d3d_cakenn_free
            const GREEN = EGGO.get_greens_function_tables(MODEL_NAME)
            const BASIS_FUNCTIONS = EGGO.get_basis_functions(MODEL_NAME)
            const BASIS_FUNCTIONS_1D, BF1D_ITP = EGGO.get_basis_functions_1d(MODEL_NAME)
            const NNMODEL = EGGO.get_model(MODEL_NAME)
            const NNMODEL1D = EGGO.get_model1d(MODEL_NAME)
            
            println("✅ Models loaded and cached in Julia globals")
        )");
        
        if (jl_exception_occurred()) {
            jl_call2(jl_get_function(jl_base_module, "showerror"), 
                     jl_stderr_obj(), jl_exception_occurred());
            throw std::runtime_error("Failed to load models");
        }
        
        models_loaded = true;
        std::cout << "Models loaded successfully and cached!" << std::endl;
    }
    
    std::string vector_to_julia_string(const std::vector<double>& vec) {
        std::stringstream ss;
        ss << "[";
        for (size_t i = 0; i < vec.size(); i++) {
            if (i > 0) ss << ", ";
            ss << vec[i];
        }
        ss << "]";
        return ss.str();
    }
    
    std::pair<std::vector<double>, std::vector<double>> run_prediction(
        int shot,
        const std::vector<double>& expsi,
        const std::vector<double>& fwtsi, 
        const std::vector<double>& expmp2,
        const std::vector<double>& fwtmp2,
        const std::vector<double>& fcurrt,
        const std::vector<double>& ecurrt,
        double Ip) {
        
        // Ensure models are loaded
        if (!models_loaded) {
            load_models_once();
        }
        
        std::cout << "Running prediction for shot " << shot << " (using cached models)..." << std::endl;
        
        // Create Julia code that uses the pre-loaded models
        std::stringstream julia_code;
        julia_code << "shot = " << shot << "\n";
        julia_code << "expsi = Float64" << vector_to_julia_string(expsi) << "\n";
        julia_code << "fwtsi = Float64" << vector_to_julia_string(fwtsi) << "\n";
        julia_code << "expmp2 = Float64" << vector_to_julia_string(expmp2) << "\n";
        julia_code << "fwtmp2 = Float64" << vector_to_julia_string(fwtmp2) << "\n";
        julia_code << "fcurrt = " << vector_to_julia_string(fcurrt) << "\n";
        julia_code << "ecurrt = " << vector_to_julia_string(ecurrt) << "\n";
        julia_code << "Ip = " << Ip << "\n";
        
        // Use the pre-loaded global models (much faster!)
        julia_code << "y_psi, y1d = EGGO.predict_psipla_free(shot, expsi, fwtsi, expmp2, fwtmp2, fcurrt, ecurrt, Ip, NNMODEL, NNMODEL1D, GREEN, BASIS_FUNCTIONS)\n";
        
        std::string code_str = julia_code.str();
        jl_eval_string(code_str.c_str());
        
        if (jl_exception_occurred()) {
            jl_call2(jl_get_function(jl_base_module, "showerror"), 
                     jl_stderr_obj(), jl_exception_occurred());
            throw std::runtime_error("Julia prediction failed");
        }
        
        // Extract results
        jl_value_t* y_psi_jl = jl_eval_string("y_psi");
        jl_value_t* y1d_jl = jl_eval_string("y1d");
        
        if (jl_exception_occurred()) {
            throw std::runtime_error("Failed to extract results");
        }
        
        // Convert Julia arrays to C++ vectors
        std::vector<double> y_psi_cpp = julia_array_to_vector(y_psi_jl);
        std::vector<double> y1d_cpp = julia_array_to_vector(y1d_jl);
        
        std::cout << "✅ Prediction completed!" << std::endl;
        std::cout << "   PSI coefficients: " << y_psi_cpp.size() << " elements" << std::endl;
        std::cout << "   1D profile coefficients: " << y1d_cpp.size() << " elements" << std::endl;
        
        return std::make_pair(y_psi_cpp, y1d_cpp);
    }
    
private:
    std::vector<double> julia_array_to_vector(jl_value_t* julia_arr) {
        if (!jl_is_array(julia_arr)) {
            throw std::runtime_error("Expected Julia array");
        }
        
        jl_array_t* arr = (jl_array_t*)julia_arr;
        
        // Handle both 1D and 2D arrays (flatten 2D)
        size_t total_len = jl_array_len(arr);
        double* data = (double*)jl_array_data(arr, double);
        
        return std::vector<double>(data, data + total_len);
    }
};

int main() {
    try {
        // Initialize EGGO predictor
        OptimizedJuliaEGGO eggo;
        
        // Load models once (expensive operation)
        auto start_load = std::chrono::high_resolution_clock::now();
        eggo.load_models_once();
        auto end_load = std::chrono::high_resolution_clock::now();
        auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_load - start_load);
        std::cout << "Model loading took: " << load_time.count() << " ms" << std::endl;
        
        // Example diagnostic data
        std::vector<double> fwtmp2 = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0};
        std::vector<double> fwtsi = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        std::vector<double> expsi = {-0.4033802002319999, 0.06076110283043938, 0.10853515679092042, 0.15641445005947727, 0.1660594820489152, -0.11031321378098395, 0.5896888595615067, 0.1753863205962317, 0.088503716473891, -0.006736425708497283, 0.0380201053265609, 0.09299474023561918, 0.18626762407768788, 0.21683726509048587, -0.09977765782429186, -0.10873433915110613, 0.19441311835928618, 0.10148185329008365, 0.011709894683019983, 0.044467497616696994, 0.0923198628231168, 0.1355893842074772, 0.15601260656990892, 0.16695561530681619, 0.0916941833019791, -0.06146592144899354, -0.04386941444183142, -0.15388324682855725, -0.08224239929633187, 0.028333176401003295, 0.07246295585820689, 0.14302804900543153, 0.1965329452133967, 0.20757885694694528, 0.07730990109015949, -0.07111951762372472, -0.044894917507311526, 0.04801603600667154, -0.06804283453097908, 0.168213291249242, 0.1298883611810984, 0.16620308388694635, 0.1388520455946071, 0.10336328686838459};
        std::vector<double> expmp2 = {0.5751758217811584, 0.5777432918548584, 0.2995361089706421, 0.5352239012718201, 0.4268638491630554, 0.3233080506324768, 0.17644630372524261, 0.30761346220970154, 0.1082150787115097, 0.2665691375732422, 0.25748637318611145, 0.26021692156791687, 0.29702144861221313, 0.360135555267334, 0.3123251497745514, 0.5910444259643555, 0.39638155698776245, 0.580190896987915, 0.5035938620567322, 0.42172273993492126, 0.2821279466152191, 0.3101239502429962, 0.5534896850585938, 0.23387545347213745, 0.30140069127082825, 0.289532333612442, 0.29198628664016724, 0.3242175579071045, 0.35721850395202637, -0.05749744176864624, -0.010924776084721088, 0.03236999735236168, 0.09824628382921219, 0.12703147530555725, 0.19003534317016602, 0.2579684555530548, 0.2565344572067261, 0.2962685525417328, 0.3557198643684387, 0.3125780522823334, 0.36826983094215393, 0.3251114785671234, 0.3021714389324188, 0.30776268243789673, 0.2574252486228943, 0.13159792125225067, 0.05727966129779816, 0.027812741696834564, -0.0451042503118515, -0.12220883369445801, -0.024340426549315453, 0.24090157449245453, 0.4052409827709198, 0.5804176330566406, 0.5786687135696411, 0.5555588603019714, 0.30065643787384033, 0.13553518056869507, -0.02408536896109581, -0.036081016063690186, -0.041902780532836914, 0.09766237437725067, 0.09371349960565567, 0.08389651775360107, -0.0326533168554306, -0.024524418637156487, -0.5368168950080872, -0.14827822148799896, -0.046501222997903824, -0.05045483633875847, 0.06118004024028778, 0.020677056163549423, 0.0, 0.04194311797618866, 0.06540253013372421, 0.06637807935476303};
        std::vector<double> fcurrt = {-249457.546875, -68876.06201171875, -2278.434410095215, 141849.37939453125, 86605.60424804688, -219819.0478515625, -165385.4296875, 114270.47143554688, -21949.81216430664, -265337.65234375, -136116.3427734375, -74327.11328125, 199696.208984375, 200311.96337890625, -214281.10107421875, -206268.06030273438, 77999.93920898438, -21844.74853515625};
        std::vector<double> ecurrt = {-41790.4296875, -40929.421875, -41262.1875, -40731.24609375, -41504.9609375, -40838.37109375};
        double Ip = 1.2601925881818281e6;
        
        std::cout << "\n=== Running multiple predictions with cached models ===" << std::endl;
        
        // Run multiple predictions (fast after first model load)
        for (int i = 0; i < 3; i++) {
            int shot = 168830;
            
            auto start_pred = std::chrono::high_resolution_clock::now();
            auto [y_psi, y1d] = eggo.run_prediction(shot, expsi, fwtsi, expmp2, fwtmp2, fcurrt, ecurrt, Ip);
            auto end_pred = std::chrono::high_resolution_clock::now();
            auto pred_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_pred - start_pred);
            
            std::cout << "Prediction " << (i+1) << " took: " << pred_time.count() << " ms" << std::endl;
            std::cout << "First 3 PSI coefficients: ";
            for (size_t j = 0; j < std::min(size_t(3), y_psi.size()); j++) {
                std::cout << y_psi[j] << " ";
            }
            std::cout << "\n" << std::endl;
        }
        
        std::cout << "🎉 All predictions completed successfully!" << std::endl;
        std::cout << "Note: First model load is slow, subsequent predictions are fast!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
