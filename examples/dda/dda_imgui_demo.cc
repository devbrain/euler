/**
 * @file dda_imgui_demo.cc
 * @brief Comprehensive interactive demonstration of Euler DDA algorithms using ImGui
 */

#include <SDL.h>
#include <imgui.h>

// Handle SDL version after SDL.h is included
#ifdef SDL_VERSION
#undef SDL_VERSION
#endif

#if SDL_VERSION_MACRO == 3
#include <imgui_impl_sdl3.h>
#include <imgui_impl_sdlrenderer3.h>
#else
#include <imgui_impl_sdl2.h>
#include <imgui_impl_sdlrenderer2.h>
#endif

#include <euler/euler.hh>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

using namespace euler;
using namespace euler::dda;
using namespace euler::dda::curves;

// Constants
constexpr int WINDOW_WIDTH = 1280;
constexpr int WINDOW_HEIGHT = 800;
constexpr int GRID_SIZE = 600;
constexpr int GRID_CELL_SIZE = 10;

// Color definitions
struct Color {
    float r, g, b, a;
    
    [[nodiscard]] ImU32 toImU32() const {
        return IM_COL32(r * 255, g * 255, b * 255, a * 255);
    }
};

constexpr Color COLOR_GRID = {0.2f, 0.2f, 0.2f, 1.0f};
constexpr Color COLOR_AXES = {0.4f, 0.4f, 0.4f, 1.0f};
constexpr  Color COLOR_LINE = {0.0f, 1.0f, 0.0f, 1.0f};
constexpr  Color COLOR_CIRCLE = {1.0f, 0.0f, 0.0f, 1.0f};
constexpr  Color COLOR_ELLIPSE = {1.0f, 1.0f, 0.0f, 1.0f};
constexpr  Color COLOR_CURVE = {0.0f, 1.0f, 1.0f, 1.0f};
constexpr  Color COLOR_BEZIER = {1.0f, 0.0f, 1.0f, 1.0f};
constexpr  Color COLOR_BSPLINE = {1.0f, 0.5f, 0.0f, 1.0f};
constexpr  Color COLOR_AA_PRIMARY = {1.0f, 1.0f, 1.0f, 1.0f};
constexpr  Color COLOR_CONTROL_POINT = {0.5f, 0.5f, 1.0f, 1.0f};

// Visualization state
struct VisualizationState {
    // View settings
    ImVec2 grid_center = {GRID_SIZE / 2.0f, GRID_SIZE / 2.0f};
    float zoom = 1.0f;
    bool show_grid = true;
    bool show_axes = true;
    bool show_coordinates = false;
    bool show_pixel_info = true;
    bool animate = false;
    float animation_time = 0.0f;
    float animation_speed = 1.0f;
    
    // Algorithm selection
    enum AlgorithmType {
        ALGO_LINE,
        ALGO_THICK_LINE,
        ALGO_AA_LINE,
        ALGO_CIRCLE,
        ALGO_ARC,
        ALGO_FILLED_CIRCLE,
        ALGO_FILLED_ARC,
        ALGO_AA_CIRCLE,
        ALGO_AA_ARC,
        ALGO_ELLIPSE,
        ALGO_ELLIPSE_ARC,
        ALGO_FILLED_ELLIPSE,
        ALGO_FILLED_ELLIPSE_ARC,
        ALGO_AA_ELLIPSE,
        ALGO_AA_ELLIPSE_ARC,
        ALGO_CURVE,
        ALGO_BEZIER,
        ALGO_BSPLINE,
        ALGO_MATH_CURVES
    };
    AlgorithmType current_algorithm = ALGO_LINE;
    
    // Line parameters
    point2f line_start = {10, 10};
    point2f line_end = {40, 30};
    float line_thickness = 3.0f;
    
    // Circle parameters
    point2f circle_center = {30, 30};
    float circle_radius = 15.0f;
    
    // Ellipse parameters
    point2f ellipse_center = {30, 30};
    float ellipse_a = 20.0f;
    float ellipse_b = 10.0f;
    float ellipse_rotation = 0.0f;
    
    // Arc parameters
    float arc_start_angle = 0.0f;
    float arc_end_angle = 180.0f;
    bool draw_arc = false;
    
    // Curve parameters
    int curve_type = 0; // 0: parabola, 1: sine, 2: spiral, 3: custom
    float curve_t_min = 0.0f;
    float curve_t_max = 10.0f;
    float curve_scale = 5.0f;
    
    // Bezier parameters
    std::vector<point2f> bezier_points = {{10.0f, 40.0f}, {20.0f, 10.0f}, {40.0f, 10.0f}, {50.0f, 40.0f}};
    
    // B-spline parameters
    std::vector<point2f> bspline_points = {{10, 30}, {20, 10}, {30, 20}, {40, 10}, {50, 30}};
    int bspline_degree = 3;
    
    // Mathematical curves
    enum MathCurveType {
        MATH_CARDIOID,
        MATH_LIMACON,
        MATH_ROSE,
        MATH_LEMNISCATE,
        MATH_ASTROID,
        MATH_EPICYCLOID,
        MATH_HYPOCYCLOID,
        MATH_LISSAJOUS
    };
    MathCurveType math_curve_type = MATH_CARDIOID;
    float math_param_a = 10.0f;
    float math_param_b = 5.0f;
    int math_param_n = 5;
    
    // Antialiasing parameters
    aa_algorithm aa_algo = aa_algorithm::wu;
    
    // Performance options
    bool use_batch_pixels = false;
    
    // Rendered pixels
    std::vector<pixel<int>> pixels;
    std::vector<span> spans;
    std::vector<aa_pixel<float>> aa_pixels;
    
    // Batch storage
    std::vector<pixel_batch<pixel<int>>> pixel_batches;
    std::vector<pixel_batch<aa_pixel<float>>> aa_pixel_batches;
};

class DDADemo {
private:
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    VisualizationState state;
    ImVec2 canvas_pos;
    ImVec2 canvas_size;
    
public:
    DDADemo() = default;
    ~DDADemo() {
        cleanup();
    }
    
    bool init() {
        // Initialize SDL
        if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
            return false;
        }
        
        // Create window
        window = SDL_CreateWindow("Euler DDA Algorithm Showcase",
                                  SDL_WINDOWPOS_CENTERED,
                                  SDL_WINDOWPOS_CENTERED,
                                  WINDOW_WIDTH, WINDOW_HEIGHT,
                                  SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
        if (!window) {
            return false;
        }
        
        // Create renderer
        renderer = SDL_CreateRenderer(window, -1, 
                                      SDL_RENDERER_PRESENTVSYNC | SDL_RENDERER_ACCELERATED);
        if (!renderer) {
            return false;
        }
        
        // Setup ImGui
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        
        ImGui::StyleColorsDark();
        
#if SDL_VERSION_MACRO == 3
        ImGui_ImplSDL3_InitForSDLRenderer(window, renderer);
        ImGui_ImplSDLRenderer3_Init(renderer);
#else
        ImGui_ImplSDL2_InitForSDLRenderer(window, renderer);
        ImGui_ImplSDLRenderer2_Init(renderer);
#endif
        
        // Initialize pixels for the default algorithm
        updatePixels();
        
        return true;
    }
    
    void cleanup() {
#if SDL_VERSION_MACRO == 3
        ImGui_ImplSDLRenderer3_Shutdown();
        ImGui_ImplSDL3_Shutdown();
#else
        ImGui_ImplSDLRenderer2_Shutdown();
        ImGui_ImplSDL2_Shutdown();
#endif
        ImGui::DestroyContext();
        
        if (renderer) {
            SDL_DestroyRenderer(renderer);
            renderer = nullptr;
        }
        
        if (window) {
            SDL_DestroyWindow(window);
            window = nullptr;
        }
        
        SDL_Quit();
    }
    
    void run() {
        bool running = true;
        
        while (running) {
            SDL_Event event;
            while (SDL_PollEvent(&event)) {
#if SDL_VERSION_MACRO == 3
                ImGui_ImplSDL3_ProcessEvent(&event);
#else
                ImGui_ImplSDL2_ProcessEvent(&event);
#endif
                
                if (event.type == SDL_QUIT) {
                    running = false;
                }
            }
            
            // Start ImGui frame
#if SDL_VERSION_MACRO == 3
            ImGui_ImplSDLRenderer3_NewFrame();
            ImGui_ImplSDL3_NewFrame();
#else
            ImGui_ImplSDLRenderer2_NewFrame();
            ImGui_ImplSDL2_NewFrame();
#endif
            ImGui::NewFrame();
            
            // Update
            update();
            
            // Render
            render();
            
            // Present
            SDL_RenderPresent(renderer);
        }
    }
    
private:
    void update() {
        // Update animation
        if (state.animate) {
            state.animation_time += ImGui::GetIO().DeltaTime * state.animation_speed;
            updatePixels();
        }
        
        // Main menu bar
        if (ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem("Reset View")) {
                    state.grid_center = {GRID_SIZE / 2.0f, GRID_SIZE / 2.0f};
                    state.zoom = 1.0f;
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Exit")) {
                    SDL_Event quit_event;
                    quit_event.type = SDL_QUIT;
                    SDL_PushEvent(&quit_event);
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("View")) {
                ImGui::MenuItem("Show Grid", nullptr, &state.show_grid);
                ImGui::MenuItem("Show Axes", nullptr, &state.show_axes);
                ImGui::MenuItem("Show Coordinates", nullptr, &state.show_coordinates);
                ImGui::MenuItem("Show Pixel Info", nullptr, &state.show_pixel_info);
                ImGui::EndMenu();
            }
            ImGui::EndMainMenuBar();
        }
        
        // Control panel
        ImGui::SetNextWindowPos(ImVec2(10, 30), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(350, 760), ImGuiCond_FirstUseEver);
        ImGui::Begin("DDA Controls");
        
        // Algorithm selection
        ImGui::Text("Algorithm:");
        const char* algo_names[] = {
            "Line", "Thick Line", "Antialiased Line",
            "Circle", "Arc", "Filled Circle", "Filled Arc", 
            "Antialiased Circle", "Antialiased Arc",
            "Ellipse", "Ellipse Arc", "Filled Ellipse", "Filled Ellipse Arc",
            "Antialiased Ellipse", "Antialiased Ellipse Arc",
            "Parametric Curve", "Bezier Curve", "B-Spline",
            "Mathematical Curves"
        };
        int algo_idx = (int)state.current_algorithm;
        if (ImGui::Combo("##Algorithm", &algo_idx, algo_names, IM_ARRAYSIZE(algo_names))) {
            state.current_algorithm = (VisualizationState::AlgorithmType)algo_idx;
            updatePixels();
        }
        
        ImGui::Separator();
        
        // Algorithm-specific controls
        switch (state.current_algorithm) {
            case VisualizationState::ALGO_LINE:
            case VisualizationState::ALGO_AA_LINE:
                drawLineControls();
                break;
            case VisualizationState::ALGO_THICK_LINE:
                drawLineControls();
                ImGui::SliderFloat("Thickness", &state.line_thickness, 1.0f, 20.0f);
                break;
            case VisualizationState::ALGO_CIRCLE:
            case VisualizationState::ALGO_FILLED_CIRCLE:
            case VisualizationState::ALGO_AA_CIRCLE:
                drawCircleControls();
                break;
            case VisualizationState::ALGO_ARC:
            case VisualizationState::ALGO_FILLED_ARC:
            case VisualizationState::ALGO_AA_ARC:
                drawCircleControls();
                drawArcControls();
                break;
            case VisualizationState::ALGO_ELLIPSE:
            case VisualizationState::ALGO_FILLED_ELLIPSE:
            case VisualizationState::ALGO_AA_ELLIPSE:
                drawEllipseControls();
                break;
            case VisualizationState::ALGO_ELLIPSE_ARC:
            case VisualizationState::ALGO_FILLED_ELLIPSE_ARC:
            case VisualizationState::ALGO_AA_ELLIPSE_ARC:
                drawEllipseControls();
                drawArcControls();
                break;
            case VisualizationState::ALGO_CURVE:
                drawCurveControls();
                break;
            case VisualizationState::ALGO_BEZIER:
                drawBezierControls();
                break;
            case VisualizationState::ALGO_BSPLINE:
                drawBSplineControls();
                break;
            case VisualizationState::ALGO_MATH_CURVES:
                drawMathCurveControls();
                break;
        }
        
        // Antialiasing options
        if (state.current_algorithm == VisualizationState::ALGO_AA_LINE ||
            state.current_algorithm == VisualizationState::ALGO_AA_CIRCLE ||
            state.current_algorithm == VisualizationState::ALGO_AA_ELLIPSE) {
            ImGui::Separator();
            ImGui::Text("Antialiasing Algorithm:");
            const char* aa_names[] = {"Wu", "Gupta-Sproull", "Supersampling"};
            int aa_idx = (int)state.aa_algo;
            if (ImGui::Combo("##AA", &aa_idx, aa_names, IM_ARRAYSIZE(aa_names))) {
                state.aa_algo = (aa_algorithm)aa_idx;
                updatePixels();
            }
        }
        
        ImGui::Separator();
        
        // Animation controls
        ImGui::Checkbox("Animate", &state.animate);
        if (state.animate) {
            ImGui::SliderFloat("Speed", &state.animation_speed, 0.1f, 5.0f);
        }
        
        // Performance options
        ImGui::Separator();
        ImGui::Text("Performance:");
        bool batch_changed = ImGui::Checkbox("Use Batch Pixels", &state.use_batch_pixels);
        if (batch_changed) updatePixels();
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Use batched iterators for better cache performance");
        }
        
        // Statistics
        ImGui::Separator();
        ImGui::Text("Statistics:");
        int pixel_count = state.pixels.size() + state.aa_pixels.size();
        int span_pixels = 0;
        for (const auto& span : state.spans) {
            span_pixels += span.x_end - span.x_start + 1;
        }
        pixel_count += span_pixels;
        ImGui::Text("Pixels rendered: %d", pixel_count);
        
        if (state.use_batch_pixels && !state.pixel_batches.empty()) {
            ImGui::Text("Pixel batches: %zu", state.pixel_batches.size());
            
            // Calculate average batch size
            size_t total_batch_pixels = 0;
            for (const auto& batch : state.pixel_batches) {
                total_batch_pixels += batch.size();
            }
            float avg_batch_size = static_cast<float>(total_batch_pixels) / state.pixel_batches.size();
            ImGui::Text("Avg batch size: %.1f", avg_batch_size);
        }
        
        ImGui::End();
        
        // Canvas window
        ImGui::SetNextWindowPos(ImVec2(370, 30), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(900, 760), ImGuiCond_FirstUseEver);
        ImGui::Begin("Canvas", nullptr, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
        
        drawCanvas();
        
        ImGui::End();
    }
    
    void drawLineControls() {
        ImGui::Text("Line Parameters:");
        bool changed = false;
        changed |= ImGui::DragFloat2("Start", &state.line_start.x, 0.5f);
        changed |= ImGui::DragFloat2("End", &state.line_end.x, 0.5f);
        if (changed) updatePixels();
    }
    
    void drawCircleControls() {
        ImGui::Text("Circle Parameters:");
        bool changed = false;
        changed |= ImGui::DragFloat2("Center", &state.circle_center.x, 0.5f);
        changed |= ImGui::DragFloat("Radius", &state.circle_radius, 0.5f, 0.1f, 100.0f);
        
        if (changed) updatePixels();
    }
    
    void drawEllipseControls() {
        ImGui::Text("Ellipse Parameters:");
        bool changed = false;
        changed |= ImGui::DragFloat2("Center", &state.ellipse_center.x, 0.5f);
        changed |= ImGui::DragFloat("Semi-major (a)", &state.ellipse_a, 0.5f, 0.1f, 100.0f);
        changed |= ImGui::DragFloat("Semi-minor (b)", &state.ellipse_b, 0.5f, 0.1f, 100.0f);
        
        if (changed) updatePixels();
    }
    
    void drawArcControls() {
        ImGui::Text("Arc Parameters:");
        bool changed = false;
        changed |= ImGui::SliderFloat("Start Angle", &state.arc_start_angle, 0.0f, 360.0f);
        changed |= ImGui::SliderFloat("End Angle", &state.arc_end_angle, 0.0f, 360.0f);
        
        if (changed) updatePixels();
    }
    
    void drawCurveControls() {
        ImGui::Text("Curve Parameters:");
        bool changed = false;
        
        const char* curve_types[] = {"Parabola", "Sine Wave", "Spiral", "Lissajous"};
        changed |= ImGui::Combo("Type", &state.curve_type, curve_types, IM_ARRAYSIZE(curve_types));
        
        changed |= ImGui::DragFloat("T Min", &state.curve_t_min, 0.1f);
        changed |= ImGui::DragFloat("T Max", &state.curve_t_max, 0.1f);
        changed |= ImGui::DragFloat("Scale", &state.curve_scale, 0.5f, 0.1f, 50.0f);
        
        if (changed) updatePixels();
    }
    
    void drawBezierControls() {
        ImGui::Text("Bezier Curve:");
        bool changed = false;
        
        for (size_t i = 0; i < state.bezier_points.size(); ++i) {
            ImGui::PushID(i);
            changed |= ImGui::DragFloat2("", &state.bezier_points[i].x, 0.5f);
            ImGui::SameLine();
            if (ImGui::Button("X") && state.bezier_points.size() > 2) {
                state.bezier_points.erase(state.bezier_points.begin() + i);
                changed = true;
            }
            ImGui::PopID();
        }
        
        if (ImGui::Button("Add Point")) {
            auto last = state.bezier_points.back();
            state.bezier_points.push_back({last.x + 10, last.y});
            changed = true;
        }
        
        if (changed) updatePixels();
    }
    
    void drawBSplineControls() {
        ImGui::Text("B-Spline:");
        bool changed = false;
        
        changed |= ImGui::SliderInt("Degree", &state.bspline_degree, 1, 5);
        
        for (size_t i = 0; i < state.bspline_points.size(); ++i) {
            ImGui::PushID(i);
            changed |= ImGui::DragFloat2("", &state.bspline_points[i].x, 0.5f);
            ImGui::SameLine();
            if (ImGui::Button("X") && state.bspline_points.size() > state.bspline_degree + 1) {
                state.bspline_points.erase(state.bspline_points.begin() + i);
                changed = true;
            }
            ImGui::PopID();
        }
        
        if (ImGui::Button("Add Point")) {
            auto last = state.bspline_points.back();
            state.bspline_points.push_back({last.x + 10, last.y});
            changed = true;
        }
        
        if (changed) updatePixels();
    }
    
    void drawMathCurveControls() {
        ImGui::Text("Mathematical Curves:");
        bool changed = false;
        
        const char* curve_names[] = {
            "Cardioid", "Limaçon", "Rose", "Lemniscate",
            "Astroid", "Epicycloid", "Hypocycloid", "Lissajous"
        };
        int curve_idx = (int)state.math_curve_type;
        changed |= ImGui::Combo("Curve", &curve_idx, curve_names, IM_ARRAYSIZE(curve_names));
        state.math_curve_type = (VisualizationState::MathCurveType)curve_idx;
        
        // Curve-specific parameters
        switch (state.math_curve_type) {
            case VisualizationState::MATH_CARDIOID:
            case VisualizationState::MATH_LIMACON:
                changed |= ImGui::DragFloat("a", &state.math_param_a, 0.5f, 0.1f, 50.0f);
                if (state.math_curve_type == VisualizationState::MATH_LIMACON) {
                    changed |= ImGui::DragFloat("b", &state.math_param_b, 0.5f, 0.1f, 50.0f);
                }
                break;
            case VisualizationState::MATH_ROSE:
                changed |= ImGui::DragFloat("a", &state.math_param_a, 0.5f, 0.1f, 50.0f);
                changed |= ImGui::SliderInt("n petals", &state.math_param_n, 2, 12);
                break;
            case VisualizationState::MATH_LEMNISCATE:
                changed |= ImGui::DragFloat("a", &state.math_param_a, 0.5f, 0.1f, 50.0f);
                break;
            case VisualizationState::MATH_ASTROID:
                changed |= ImGui::DragFloat("a", &state.math_param_a, 0.5f, 0.1f, 50.0f);
                break;
            case VisualizationState::MATH_EPICYCLOID:
            case VisualizationState::MATH_HYPOCYCLOID:
                changed |= ImGui::DragFloat("R", &state.math_param_a, 0.5f, 0.1f, 50.0f);
                changed |= ImGui::DragFloat("r", &state.math_param_b, 0.5f, 0.1f, 50.0f);
                break;
            case VisualizationState::MATH_LISSAJOUS:
                changed |= ImGui::DragFloat("A", &state.math_param_a, 0.5f, 0.1f, 50.0f);
                changed |= ImGui::DragFloat("B", &state.math_param_b, 0.5f, 0.1f, 50.0f);
                changed |= ImGui::SliderInt("n", &state.math_param_n, 1, 10);
                break;
        }
        
        if (changed) updatePixels();
    }
    
    void updatePixels() {
        state.pixels.clear();
        state.spans.clear();
        state.aa_pixels.clear();
        state.pixel_batches.clear();
        state.aa_pixel_batches.clear();
        
        
        switch (state.current_algorithm) {
            case VisualizationState::ALGO_LINE:
                updateLinePixels();
                break;
            case VisualizationState::ALGO_THICK_LINE:
                updateThickLinePixels();
                break;
            case VisualizationState::ALGO_AA_LINE:
                updateAALinePixels();
                break;
            case VisualizationState::ALGO_CIRCLE:
                updateCirclePixels();
                break;
            case VisualizationState::ALGO_ARC:
                updateArcPixels();
                break;
            case VisualizationState::ALGO_FILLED_CIRCLE:
                updateFilledCirclePixels();
                break;
            case VisualizationState::ALGO_FILLED_ARC:
                updateFilledArcPixels();
                break;
            case VisualizationState::ALGO_AA_CIRCLE:
                updateAACirclePixels();
                break;
            case VisualizationState::ALGO_AA_ARC:
                updateAAArcPixels();
                break;
            case VisualizationState::ALGO_ELLIPSE:
                updateEllipsePixels();
                break;
            case VisualizationState::ALGO_ELLIPSE_ARC:
                updateEllipseArcPixels();
                break;
            case VisualizationState::ALGO_FILLED_ELLIPSE:
                updateFilledEllipsePixels();
                break;
            case VisualizationState::ALGO_FILLED_ELLIPSE_ARC:
                updateFilledEllipseArcPixels();
                break;
            case VisualizationState::ALGO_AA_ELLIPSE:
                updateAAEllipsePixels();
                break;
            case VisualizationState::ALGO_AA_ELLIPSE_ARC:
                updateAAEllipseArcPixels();
                break;
            case VisualizationState::ALGO_CURVE:
                updateCurvePixels();
                break;
            case VisualizationState::ALGO_BEZIER:
                updateBezierPixels();
                break;
            case VisualizationState::ALGO_BSPLINE:
                updateBSplinePixels();
                break;
            case VisualizationState::ALGO_MATH_CURVES:
                updateMathCurvePixels();
                break;
        }
    }
    
    void updateLinePixels() {
        point2f start = state.line_start;
        point2f end = state.line_end;
        
        if (state.animate) {
            float t = std::fmod(state.animation_time, 2.0f);
            if (t > 1.0f) t = 2.0f - t;
            auto diff = end - start;
            end = point2f{start.x + diff[0] * t, start.y + diff[1] * t};
        }
        
        if (state.use_batch_pixels) {
            // Use batched line iterator
            auto line = make_batched_line(start, end);
            while (!line.at_end()) {
                // Get batch of pixels
                const auto& batch = line.current_batch();
                // Copy batch pixels to our pixel storage
                for (const auto& pixel : batch) {
                    state.pixels.push_back(pixel);
                }
                // Also store the batch for statistics
                state.pixel_batches.push_back(batch);
                line.next_batch();
            }
        } else {
            // Use regular line iterator
            auto line = make_line_iterator(start, end);
            for (; line != decltype(line)::end(); ++line) {
                state.pixels.push_back(*line);
            }
        }
    }
    
    void updateThickLinePixels() {
        point2f start = state.line_start;
        point2f end = state.line_end;
        
        if (state.animate) {
            float t = std::fmod(state.animation_time, 2.0f);
            if (t > 1.0f) t = 2.0f - t;
            auto diff = end - start;
            end = point2f{start.x + diff[0] * t, start.y + diff[1] * t};
        }
        
        auto line = make_thick_line_iterator(start, end, state.line_thickness);
        for (; line != decltype(line)::end(); ++line) {
            state.pixels.push_back(*line);
        }
    }
    
    void updateAALinePixels() {
        point2f start = state.line_start;
        point2f end = state.line_end;
        
        if (state.animate) {
            float t = std::fmod(state.animation_time, 2.0f);
            if (t > 1.0f) t = 2.0f - t;
            auto diff = end - start;
            end = point2f{start.x + diff[0] * t, start.y + diff[1] * t};
        }
        
        if (state.aa_algo == aa_algorithm::gupta_sproull) {
            auto line = make_gupta_sproull_line_iterator(start, end);
            for (; line != decltype(line)::end(); ++line) {
                state.aa_pixels.push_back(*line);
            }
        } else {
            auto line = make_aa_line_iterator(start, end);
            for (; line != decltype(line)::end(); ++line) {
                state.aa_pixels.push_back(*line);
            }
        }
    }
    
    void updateCirclePixels() {
        float radius = state.circle_radius;
        
        if (state.animate) {
            radius *= (std::sin(state.animation_time) + 1.0f) * 0.5f;
        }
        
        if (state.draw_arc) {
            auto arc = circle_iterator<float>(state.circle_center, radius,
                                             degree<float>(state.arc_start_angle),
                                             degree<float>(state.arc_end_angle));
            for (; arc != decltype(arc)::end(); ++arc) {
                state.pixels.push_back(*arc);
            }
        } else {
            auto circle = make_circle_iterator(state.circle_center, radius);
            for (; circle != decltype(circle)::end(); ++circle) {
                state.pixels.push_back(*circle);
            }
        }
    }
    
    void updateFilledCirclePixels() {
        float radius = state.circle_radius;
        
        if (state.animate) {
            radius *= (std::sin(state.animation_time) + 1.0f) * 0.5f;
        }
        
        auto filled = make_filled_circle_iterator(state.circle_center, radius);
        for (; filled != decltype(filled)::end(); ++filled) {
            state.spans.push_back(*filled);
        }
    }
    
    void updateAACirclePixels() {
        float radius = state.circle_radius;
        
        if (state.animate) {
            radius *= (std::sin(state.animation_time) + 1.0f) * 0.5f;
        }
        
        // AA circle - convert regular circle pixels to AA pixels
        auto circle = make_circle_iterator(state.circle_center, radius);
        for (; circle != decltype(circle)::end(); ++circle) {
            aa_pixel<float> p;
            p.pos = point2<float>(static_cast<float>((*circle).pos.x), static_cast<float>((*circle).pos.y));
            p.coverage = 1.0f;
            p.distance = 0.0f;
            state.aa_pixels.push_back(p);
        }
    }
    
    void updateEllipsePixels() {
        float a = state.ellipse_a;
        float b = state.ellipse_b;
        
        if (state.animate) {
            float t = std::sin(state.animation_time);
            a *= 1.0f + t * 0.3f;
            b *= 1.0f - t * 0.3f;
        }
        
        if (state.draw_arc) {
            auto arc = make_ellipse_arc_iterator(state.ellipse_center, a, b,
                                                 degree<float>(state.arc_start_angle),
                                                 degree<float>(state.arc_end_angle));
            for (; arc != decltype(arc)::end(); ++arc) {
                state.pixels.push_back(*arc);
            }
        } else {
            auto ellipse = make_ellipse_iterator(state.ellipse_center, a, b);
            for (; ellipse != decltype(ellipse)::end(); ++ellipse) {
                state.pixels.push_back(*ellipse);
            }
        }
    }
    
    void updateFilledEllipsePixels() {
        float a = state.ellipse_a;
        float b = state.ellipse_b;
        
        if (state.animate) {
            float t = std::sin(state.animation_time);
            a *= 1.0f + t * 0.3f;
            b *= 1.0f - t * 0.3f;
        }
        
        auto filled = make_filled_ellipse_iterator(state.ellipse_center, a, b);
        for (; filled != decltype(filled)::end(); ++filled) {
            state.spans.push_back(*filled);
        }
    }
    
    void updateAAEllipsePixels() {
        float a = state.ellipse_a;
        float b = state.ellipse_b;
        
        if (state.animate) {
            float t = std::sin(state.animation_time);
            a *= 1.0f + t * 0.3f;
            b *= 1.0f - t * 0.3f;
        }
        
        auto aa_ellipse = make_aa_ellipse_iterator(state.ellipse_center, a, b);
        for (; aa_ellipse != decltype(aa_ellipse)::end(); ++aa_ellipse) {
            state.aa_pixels.push_back(*aa_ellipse);
        }
    }
    
    void updateArcPixels() {
        float radius = state.circle_radius;
        
        if (state.animate) {
            radius *= (std::sin(state.animation_time) + 1.0f) * 0.5f;
        }
        
        auto arc = make_arc_iterator(state.circle_center, radius,
                                    degree<float>(state.arc_start_angle),
                                    degree<float>(state.arc_end_angle));
        for (; arc != decltype(arc)::end(); ++arc) {
            state.pixels.push_back(*arc);
        }
    }
    
    void updateFilledArcPixels() {
        float radius = state.circle_radius;
        
        if (state.animate) {
            radius *= (std::sin(state.animation_time) + 1.0f) * 0.5f;
        }
        
        auto filled = make_filled_arc_iterator(state.circle_center, radius,
                                              degree<float>(state.arc_start_angle),
                                              degree<float>(state.arc_end_angle));
        for (; filled != decltype(filled)::end(); ++filled) {
            state.spans.push_back(*filled);
        }
    }
    
    void updateAAArcPixels() {
        float radius = state.circle_radius;
        
        if (state.animate) {
            radius *= (std::sin(state.animation_time) + 1.0f) * 0.5f;
        }
        
        auto aa_arc = make_aa_arc_iterator(state.circle_center, radius,
                                          degree<float>(state.arc_start_angle),
                                          degree<float>(state.arc_end_angle));
        for (; aa_arc != decltype(aa_arc)::end(); ++aa_arc) {
            state.aa_pixels.push_back(*aa_arc);
        }
    }
    
    void updateEllipseArcPixels() {
        float a = state.ellipse_a;
        float b = state.ellipse_b;
        
        if (state.animate) {
            float t = std::sin(state.animation_time);
            a *= 1.0f + t * 0.3f;
            b *= 1.0f - t * 0.3f;
        }
        
        auto arc = make_ellipse_arc_iterator(state.ellipse_center, a, b,
                                            degree<float>(state.arc_start_angle),
                                            degree<float>(state.arc_end_angle));
        for (; arc != decltype(arc)::end(); ++arc) {
            state.pixels.push_back(*arc);
        }
    }
    
    void updateFilledEllipseArcPixels() {
        float a = state.ellipse_a;
        float b = state.ellipse_b;
        
        if (state.animate) {
            float t = std::sin(state.animation_time);
            a *= 1.0f + t * 0.3f;
            b *= 1.0f - t * 0.3f;
        }
        
        auto filled = make_filled_ellipse_arc_iterator(state.ellipse_center, a, b,
                                                      degree<float>(state.arc_start_angle),
                                                      degree<float>(state.arc_end_angle));
        for (; filled != decltype(filled)::end(); ++filled) {
            state.spans.push_back(*filled);
        }
    }
    
    void updateAAEllipseArcPixels() {
        float a = state.ellipse_a;
        float b = state.ellipse_b;
        
        if (state.animate) {
            float t = std::sin(state.animation_time);
            a *= 1.0f + t * 0.3f;
            b *= 1.0f - t * 0.3f;
        }
        
        auto aa_arc = make_aa_ellipse_arc_iterator(state.ellipse_center, a, b,
                                                   degree<float>(state.arc_start_angle),
                                                   degree<float>(state.arc_end_angle));
        for (; aa_arc != decltype(aa_arc)::end(); ++aa_arc) {
            state.aa_pixels.push_back(*aa_arc);
        }
    }
    
    void updateCurvePixels() {
        float t_min = state.curve_t_min;
        float t_max = state.curve_t_max;
        
        if (state.animate) {
            float t = std::fmod(state.animation_time * 0.5f, 1.0f);
            t_max = t_min + (t_max - t_min) * t;
        }
        
        auto curve_func = [this](float t) -> point2f {
            switch (state.curve_type) {
                case 0: // Parabola
                    return {state.grid_center.x / GRID_CELL_SIZE + t * state.curve_scale,
                            state.grid_center.y / GRID_CELL_SIZE - 0.1f * t * t * state.curve_scale};
                case 1: // Sine wave
                    return {state.grid_center.x / GRID_CELL_SIZE + t * state.curve_scale,
                            state.grid_center.y / GRID_CELL_SIZE + std::sin(t) * state.curve_scale};
                case 2: // Spiral
                    return {state.grid_center.x / GRID_CELL_SIZE + t * std::cos(t) * state.curve_scale,
                            state.grid_center.y / GRID_CELL_SIZE + t * std::sin(t) * state.curve_scale};
                case 3: // Lissajous
                    return {state.grid_center.x / GRID_CELL_SIZE + std::sin(3 * t) * state.curve_scale,
                            state.grid_center.y / GRID_CELL_SIZE + std::sin(2 * t) * state.curve_scale};
                default:
                    return {0, 0};
            }
        };
        
        auto curve = make_curve_iterator(curve_func, t_min, t_max);
        for (; curve != decltype(curve)::end(); ++curve) {
            state.pixels.push_back(*curve);
        }
    }
    
    void updateBezierPixels() {
        if (state.bezier_points.size() < 2) return;
        
        std::vector<point2f> points = state.bezier_points;
        
        if (state.animate && points.size() >= 3) {
            float t = (std::sin(state.animation_time) + 1.0f) * 0.5f;
            auto diff = points[2] - points[0];
            points[1] = point2f{points[0].x + diff[0] * t, points[0].y + diff[1] * t};
        }
        
        if (state.use_batch_pixels) {
            // Use batched Bezier iterators
            if (points.size() == 2) {
                auto line = make_batched_line(points[0], points[1]);
                while (!line.at_end()) {
                    const auto& batch = line.current_batch();
                    for (const auto& pixel : batch) {
                        state.pixels.push_back(pixel);
                    }
                    state.pixel_batches.push_back(batch);
                    line.next_batch();
                }
            } else if (points.size() == 4) {
                // Use batched cubic bezier
                auto bezier = make_batched_cubic_bezier(points[0], points[1], points[2], points[3]);
                while (!bezier.at_end()) {
                    const auto& batch = bezier.current_batch();
                    for (const auto& pixel : batch) {
                        state.pixels.push_back(pixel);
                    }
                    state.pixel_batches.push_back(batch);
                    bezier.next_batch();
                }
            } else {
                // Use batched general bezier
                auto bezier = make_batched_bezier(points);
                while (!bezier.at_end()) {
                    const auto& batch = bezier.current_batch();
                    for (const auto& pixel : batch) {
                        state.pixels.push_back(pixel);
                    }
                    state.pixel_batches.push_back(batch);
                    bezier.next_batch();
                }
            }
        } else {
            // Use regular iterators
            if (points.size() == 2) {
                auto line = make_line_iterator(points[0], points[1]);
                for (; line != decltype(line)::end(); ++line) {
                    state.pixels.push_back(*line);
                }
            } else if (points.size() == 3) {
                auto bezier = make_quadratic_bezier(points[0], points[1], points[2]);
                for (; bezier != decltype(bezier)::end(); ++bezier) {
                    state.pixels.push_back(*bezier);
                }
            } else if (points.size() == 4) {
                auto bezier = make_cubic_bezier(points[0], points[1], points[2], points[3]);
                for (; bezier != decltype(bezier)::end(); ++bezier) {
                    state.pixels.push_back(*bezier);
                }
            } else {
                auto bezier = make_bezier(points);
                for (; bezier != decltype(bezier)::end(); ++bezier) {
                    state.pixels.push_back(*bezier);
                }
            }
        }
    }
    
    void updateBSplinePixels() {
        if (state.bspline_points.size() < state.bspline_degree + 1) return;
        
        std::vector<point2f> points = state.bspline_points;
        
        if (state.animate && points.size() >= 3) {
            float t = std::sin(state.animation_time * 2.0f) * 0.2f;
            for (size_t i = 1; i < points.size() - 1; ++i) {
                points[i].y = state.bspline_points[i].y + t * 10.0f * std::sin(i * 0.5f);
            }
        }
        
        auto spline = make_bspline(points, state.bspline_degree);
        for (; spline != decltype(spline)::end(); ++spline) {
            state.pixels.push_back(*spline);
        }
    }
    
    void updateMathCurvePixels() {
        point2f center = {state.grid_center.x / GRID_CELL_SIZE, state.grid_center.y / GRID_CELL_SIZE};
        
        switch (state.math_curve_type) {
            case VisualizationState::MATH_CARDIOID: {
                auto curve = make_polar_curve(cardioid(state.math_param_a), 0.0f, 2.0f * pi, center);
                for (; curve != decltype(curve)::end(); ++curve) {
                    state.pixels.push_back(*curve);
                }
                break;
            }
            case VisualizationState::MATH_LIMACON: {
                auto curve = make_polar_curve(limacon(state.math_param_a, state.math_param_b), 0.0f, 2.0f * pi, center);
                for (; curve != decltype(curve)::end(); ++curve) {
                    state.pixels.push_back(*curve);
                }
                break;
            }
            case VisualizationState::MATH_ROSE: {
                auto curve = make_polar_curve(rose(state.math_param_a, static_cast<float>(state.math_param_n)), 0.0f, 2.0f * pi, center);
                for (; curve != decltype(curve)::end(); ++curve) {
                    state.pixels.push_back(*curve);
                }
                break;
            }
            case VisualizationState::MATH_LEMNISCATE: {
                auto curve = make_polar_curve(lemniscate(state.math_param_a), 0.0f, 2.0f * pi, center);
                for (; curve != decltype(curve)::end(); ++curve) {
                    state.pixels.push_back(*curve);
                }
                break;
            }
            case VisualizationState::MATH_ASTROID: {
                auto curve = make_curve_iterator(astroid(state.math_param_a, center), 0.0f, 2.0f * pi);
                for (; curve != decltype(curve)::end(); ++curve) {
                    state.pixels.push_back(*curve);
                }
                break;
            }
            case VisualizationState::MATH_EPICYCLOID: {
                auto curve = make_curve_iterator(epicycloid(state.math_param_a, state.math_param_b, center), 0.0f, 10.0f * pi);
                for (; curve != decltype(curve)::end(); ++curve) {
                    state.pixels.push_back(*curve);
                }
                break;
            }
            case VisualizationState::MATH_HYPOCYCLOID: {
                auto curve = make_curve_iterator(hypocycloid(state.math_param_a, state.math_param_b, center), 0.0f, 10.0f * pi);
                for (; curve != decltype(curve)::end(); ++curve) {
                    state.pixels.push_back(*curve);
                }
                break;
            }
            case VisualizationState::MATH_LISSAJOUS: {
                // Lissajous curve: x = A*sin(a*t + δ), y = B*sin(b*t)
                auto lissajous = [=](float t) -> point2f {
                    return {center.x + state.math_param_a * std::sin(state.math_param_n * t),
                            center.y + state.math_param_b * std::sin((state.math_param_n + 1) * t)};
                };
                auto curve = make_curve_iterator(lissajous, 0.0f, 2.0f * pi);
                for (; curve != decltype(curve)::end(); ++curve) {
                    state.pixels.push_back(*curve);
                }
                break;
            }
        }
    }
    
    void drawCanvas() {
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        canvas_pos = ImGui::GetCursorScreenPos();
        canvas_size = ImGui::GetContentRegionAvail();
        
        // Draw background
        draw_list->AddRectFilled(canvas_pos, 
                                 ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y),
                                 IM_COL32(10, 10, 10, 255));
        
        // Handle mouse input
        handleCanvasInput();
        
        // Set up clipping
        draw_list->PushClipRect(canvas_pos, 
                               ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y), 
                               true);
        
        // Draw grid
        if (state.show_grid) {
            drawGrid(draw_list);
        }
        
        // Draw axes
        if (state.show_axes) {
            drawAxes(draw_list);
        }
        
        // Draw pixels
        drawPixels(draw_list);
        
        // Draw control points for curves
        if (state.current_algorithm == VisualizationState::ALGO_BEZIER) {
            drawControlPoints(draw_list, state.bezier_points);
        } else if (state.current_algorithm == VisualizationState::ALGO_BSPLINE) {
            drawControlPoints(draw_list, state.bspline_points);
        }
        
        // Draw hover info
        if (state.show_pixel_info) {
            drawPixelInfo(draw_list);
        }
        
        draw_list->PopClipRect();
    }
    
    void handleCanvasInput() {
        ImGuiIO& io = ImGui::GetIO();
        
        // Pan with middle mouse button
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
            ImVec2 delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Middle);
            state.grid_center.x += delta.x;
            state.grid_center.y += delta.y;
            ImGui::ResetMouseDragDelta(ImGuiMouseButton_Middle);
        }
        
        // Zoom with mouse wheel
        if (ImGui::IsWindowHovered() && io.MouseWheel != 0) {
            float zoom_delta = io.MouseWheel * 0.1f;
            state.zoom = std::clamp(state.zoom + zoom_delta, 0.1f, 5.0f);
        }
    }
    
    void drawGrid(ImDrawList* draw_list) {
        float cell_size = GRID_CELL_SIZE * state.zoom;
        ImVec2 grid_start = {
            canvas_pos.x + state.grid_center.x - std::floor(state.grid_center.x / cell_size) * cell_size,
            canvas_pos.y + state.grid_center.y - std::floor(state.grid_center.y / cell_size) * cell_size
        };
        
        // Vertical lines
        for (float x = grid_start.x; x < canvas_pos.x + canvas_size.x; x += cell_size) {
            draw_list->AddLine(ImVec2(x, canvas_pos.y), 
                               ImVec2(x, canvas_pos.y + canvas_size.y), 
                               COLOR_GRID.toImU32());
        }
        
        // Horizontal lines
        for (float y = grid_start.y; y < canvas_pos.y + canvas_size.y; y += cell_size) {
            draw_list->AddLine(ImVec2(canvas_pos.x, y), 
                               ImVec2(canvas_pos.x + canvas_size.x, y), 
                               COLOR_GRID.toImU32());
        }
    }
    
    void drawAxes(ImDrawList* draw_list) const {
        ImVec2 origin = {
            canvas_pos.x + state.grid_center.x,
            canvas_pos.y + state.grid_center.y
        };
        
        // X axis
        draw_list->AddLine(ImVec2(canvas_pos.x, origin.y), 
                           ImVec2(canvas_pos.x + canvas_size.x, origin.y), 
                           COLOR_AXES.toImU32(), 2.0f);
        
        // Y axis
        draw_list->AddLine(ImVec2(origin.x, canvas_pos.y), 
                           ImVec2(origin.x, canvas_pos.y + canvas_size.y), 
                           COLOR_AXES.toImU32(), 2.0f);
    }
    
    void drawPixels(ImDrawList* draw_list) {
        float cell_size = GRID_CELL_SIZE * state.zoom;
        
        // Draw regular pixels
        for (const auto& pixel : state.pixels) {
            ImVec2 pos = {
                canvas_pos.x + state.grid_center.x + pixel.pos.x * cell_size,
                canvas_pos.y + state.grid_center.y + pixel.pos.y * cell_size
            };
            
            Color color = getColorForAlgorithm();
            draw_list->AddRectFilled(pos, 
                                     ImVec2(pos.x + cell_size, pos.y + cell_size), 
                                     color.toImU32());
        }
        
        // Draw spans
        for (const auto& span : state.spans) {
            ImVec2 start = {
                canvas_pos.x + state.grid_center.x + span.x_start * cell_size,
                canvas_pos.y + state.grid_center.y + span.y * cell_size
            };
            ImVec2 end = {
                canvas_pos.x + state.grid_center.x + (span.x_end + 1) * cell_size,
                canvas_pos.y + state.grid_center.y + (span.y + 1) * cell_size
            };
            
            Color color = getColorForAlgorithm();
            draw_list->AddRectFilled(start, end, color.toImU32());
        }
        
        // Draw AA pixels
        for (const auto& pixel : state.aa_pixels) {
            ImVec2 pos = {
                canvas_pos.x + state.grid_center.x + pixel.pos.x * cell_size,
                canvas_pos.y + state.grid_center.y + pixel.pos.y * cell_size
            };
            
            Color color = COLOR_AA_PRIMARY;
            color.a = pixel.coverage;
            draw_list->AddRectFilled(pos, 
                                     ImVec2(pos.x + cell_size, pos.y + cell_size), 
                                     color.toImU32());
        }
    }
    
    void drawControlPoints(ImDrawList* draw_list, const std::vector<point2f>& points) const {
        float cell_size = GRID_CELL_SIZE * state.zoom;
        
        // Draw control polygon
        for (size_t i = 0; i < points.size() - 1; ++i) {
            ImVec2 p1 = {
                canvas_pos.x + state.grid_center.x + points[i].x * cell_size,
                canvas_pos.y + state.grid_center.y + points[i].y * cell_size
            };
            ImVec2 p2 = {
                canvas_pos.x + state.grid_center.x + points[i + 1].x * cell_size,
                canvas_pos.y + state.grid_center.y + points[i + 1].y * cell_size
            };
            draw_list->AddLine(p1, p2, IM_COL32(100, 100, 100, 255));
        }
        
        // Draw control points
        for (size_t i = 0; i < points.size(); ++i) {
            ImVec2 pos = {
                canvas_pos.x + state.grid_center.x + points[i].x * cell_size,
                canvas_pos.y + state.grid_center.y + points[i].y * cell_size
            };
            draw_list->AddCircleFilled(pos, 5.0f, COLOR_CONTROL_POINT.toImU32());
            
            // Label
            char label[32];  // Increased buffer size to avoid truncation warning
            snprintf(label, sizeof(label), "P%zu", i);
            draw_list->AddText(ImVec2(pos.x + 8, pos.y - 8), IM_COL32(255, 255, 255, 255), label);
        }
    }
    
    void drawPixelInfo(ImDrawList* draw_list) const {
        ImGuiIO& io = ImGui::GetIO();
        ImVec2 mouse_pos = io.MousePos;
        
        if (mouse_pos.x >= canvas_pos.x && mouse_pos.y >= canvas_pos.y &&
            mouse_pos.x < canvas_pos.x + canvas_size.x &&
            mouse_pos.y < canvas_pos.y + canvas_size.y) {
            
            float cell_size = GRID_CELL_SIZE * state.zoom;
            int grid_x = (mouse_pos.x - canvas_pos.x - state.grid_center.x) / cell_size;
            int grid_y = (mouse_pos.y - canvas_pos.y - state.grid_center.y) / cell_size;
            
            // Show coordinate tooltip
            ImGui::BeginTooltip();
            ImGui::Text("Grid: (%d, %d)", grid_x, grid_y);
            
            // Check if hovering over a pixel
            for (const auto& pixel : state.pixels) {
                if (pixel.pos.x == grid_x && pixel.pos.y == grid_y) {
                    ImGui::Text("Pixel at this position");
                    break;
                }
            }
            
            for (const auto& pixel : state.aa_pixels) {
                if ((int)pixel.pos.x == grid_x && (int)pixel.pos.y == grid_y) {
                    ImGui::Text("AA Pixel: coverage=%.2f, distance=%.2f", 
                                pixel.coverage, pixel.distance);
                    break;
                }
            }
            
            ImGui::EndTooltip();
        }
    }
    
    [[nodiscard]] Color getColorForAlgorithm() const {
        switch (state.current_algorithm) {
            case VisualizationState::ALGO_LINE:
            case VisualizationState::ALGO_THICK_LINE:
            case VisualizationState::ALGO_AA_LINE:
                return COLOR_LINE;
            case VisualizationState::ALGO_CIRCLE:
            case VisualizationState::ALGO_ARC:
            case VisualizationState::ALGO_FILLED_CIRCLE:
            case VisualizationState::ALGO_FILLED_ARC:
            case VisualizationState::ALGO_AA_CIRCLE:
            case VisualizationState::ALGO_AA_ARC:
                return COLOR_CIRCLE;
            case VisualizationState::ALGO_ELLIPSE:
            case VisualizationState::ALGO_ELLIPSE_ARC:
            case VisualizationState::ALGO_FILLED_ELLIPSE:
            case VisualizationState::ALGO_FILLED_ELLIPSE_ARC:
            case VisualizationState::ALGO_AA_ELLIPSE:
            case VisualizationState::ALGO_AA_ELLIPSE_ARC:
                return COLOR_ELLIPSE;
            case VisualizationState::ALGO_CURVE:
            case VisualizationState::ALGO_MATH_CURVES:
                return COLOR_CURVE;
            case VisualizationState::ALGO_BEZIER:
                return COLOR_BEZIER;
            case VisualizationState::ALGO_BSPLINE:
                return COLOR_BSPLINE;
            default:
                return {1.0f, 1.0f, 1.0f, 1.0f};
        }
    }
    
    void render() const {
        ImGui::Render();
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);
        
#if SDL_VERSION_MACRO == 3
        ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), renderer);
#else
        ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(), renderer);
#endif
    }
};

int main(int argc, char* argv[]) {
    DDADemo demo;
    
    if (!demo.init()) {
        SDL_Log("Failed to initialize demo");
        return 1;
    }
    
    demo.run();
    
    return 0;
}