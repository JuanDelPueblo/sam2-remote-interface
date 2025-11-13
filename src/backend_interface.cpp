#include "backend_interface.hpp"

#include <curl/curl.h>

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include <thread>

// Constructor
BackendInterface::BackendInterface(const std::string& base_url)
    : base_url_(base_url), backend_pid_(-1), backend_running_(false) {}

// Destructor
BackendInterface::~BackendInterface() { stopBackend(); }

// Lifecycle management
// Launch the FastAPI backend
bool BackendInterface::launchBackend(const std::string& python_executable,
                                     const std::string& api_script_path) {
  if (backend_running_) {
    return true;
  }

  // Build command to launch FastAPI with uvicorn
  std::string command =
      python_executable +
      " -m uvicorn backend.api:app --host 0.0.0.0 --port 8000 --no-access-log";

  // On POSIX systems, use a subshell to capture the PID of the uvicorn
  // process. This keeps things simple while still allowing us to send a
  // SIGTERM in stopBackend.
  std::string full_command = command + " & echo $!";

  FILE* pipe = popen(full_command.c_str(), "r");
  if (!pipe) {
    std::cerr << "Failed to start backend process" << std::endl;
    backend_running_ = false;
    backend_pid_ = -1;
    return false;
  }

  char buffer[64];
  std::string pid_str;
  if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    pid_str = buffer;
  }
  int status = pclose(pipe);
  if (!pid_str.empty()) {
    try {
      backend_pid_ = std::stoi(pid_str);
    } catch (...) {
      backend_pid_ = -1;
    }
  } else {
    backend_pid_ = -1;
  }

  if (status == -1) {
    std::cerr << "Failed to obtain backend process status" << std::endl;
  }

  if (backend_pid_ <= 0) {
    std::cerr << "Failed to determine backend PID" << std::endl;
  }

  // Poll the /health endpoint until the backend reports ready or timeout.
  const int max_retries = 50;  // up to ~10 seconds
  const auto delay = std::chrono::milliseconds(200);
  bool ready = false;

  for (int i = 0; i < max_retries; ++i) {
    std::string resp = httpGet("/health", false);
    if (!resp.empty()) {
      ready = true;
      break;
    }
    std::this_thread::sleep_for(delay);
  }

  backend_running_ = ready;
  if (!backend_running_) {
    std::cerr << "Backend did not become healthy within timeout" << std::endl;
  }

  return backend_running_;
}

bool BackendInterface::stopBackend() {
  if (!backend_running_) {
    return true;
  }

  if (backend_pid_ > 0) {
    // Send SIGTERM to allow graceful shutdown; ignore errors for now.
    ::kill(backend_pid_, SIGTERM);
  }

  backend_running_ = false;
  backend_pid_ = -1;
  return true;
}

// Status endpoint
BackendInterface::StatusResponse BackendInterface::getStatus() {
  using nlohmann::json;

  StatusResponse result;
  std::string response = httpGet("/status");
  if (response.empty()) {
    result.status = "unreachable";
    result.device = std::nullopt;
    return result;
  }

  try {
    auto j = json::parse(response);
    result.status = j.value("status", "unknown");
    if (j.contains("device") && !j["device"].is_null()) {
      result.device = j["device"].get<std::string>();
    } else {
      result.device = std::nullopt;
    }
  } catch (const std::exception& e) {
    std::cerr << "Failed to parse /status response: " << e.what() << std::endl;
    result.status = "error";
    result.device = std::nullopt;
  }

  return result;
}

// Video masking endpoints
bool BackendInterface::videoInitState(const std::string& video_frames_dir) {
  nlohmann::json j;
  j["video_frames_dir"] = video_frames_dir;
  std::string resp = httpPost("/video/init_state", j.dump());
  return !resp.empty();
}

bool BackendInterface::videoResetState() {
  std::string resp = httpPost("/video/reset_state", "{}");
  return !resp.empty();
}

BackendInterface::VideoAddPointsOrBoxResponse
BackendInterface::videoAddNewPointsOrBox(
    const VideoAddPointsOrBoxRequest& request) {
  auto body = serializeVideoAddPointsOrBoxRequest(request);
  std::string resp = httpPost("/video/add_new_points_or_box", body);

  VideoAddPointsOrBoxResponse result;
  if (resp.empty()) {
    return result;
  }

  try {
    auto j = nlohmann::json::parse(resp);
    result.out_obj_ids = j.at("out_obj_ids").get<std::vector<int>>();
    result.out_masks =
        j.at("out_masks").get<std::vector<std::vector<std::vector<bool>>>>();
  } catch (const std::exception& e) {
    std::cerr << "Failed to parse /video/add_new_points_or_box response: "
              << e.what() << std::endl;
  }

  return result;
}

BackendInterface::VideoAddMaskResponse BackendInterface::videoAddNewMask(
    const VideoAddMaskRequest& request) {
  auto body = serializeVideoAddMaskRequest(request);
  std::string resp = httpPost("/video/add_new_mask", body);

  VideoAddMaskResponse result;
  if (resp.empty()) {
    return result;
  }

  try {
    auto j = nlohmann::json::parse(resp);
    result.frame_idx = j.at("frame_idx").get<int>();
    result.out_obj_ids = j.at("out_obj_ids").get<std::vector<int>>();
    result.out_masks =
        j.at("out_masks").get<std::vector<std::vector<std::vector<bool>>>>();
  } catch (const std::exception& e) {
    std::cerr << "Failed to parse /video/add_new_mask response: " << e.what()
              << std::endl;
  }

  return result;
}

BackendInterface::VideoPropagateResponse
BackendInterface::videoPropagateInVideo(const VideoPropagateRequest& request) {
  auto body = serializeVideoPropagateRequest(request);
  std::string resp = httpPost("/video/propagate_in_video", body);

  VideoPropagateResponse result;
  if (resp.empty()) {
    return result;
  }

  try {
    auto j = nlohmann::json::parse(resp);

    // video_segments: map<int, map<int, vector<vector<bool>>>>
    if (j.contains("video_segments")) {
      for (auto& [frame_str, obj_dict] : j["video_segments"].items()) {
        int frame_idx = std::stoi(frame_str);
        std::map<int, std::vector<std::vector<bool>>> inner_map;
        for (auto& [obj_str, mask_val] : obj_dict.items()) {
          int obj_id = std::stoi(obj_str);
          inner_map[obj_id] = mask_val.get<std::vector<std::vector<bool>>>();
        }
        result.video_segments[frame_idx] = std::move(inner_map);
      }
    }

    if (j.contains("saved_mask_paths")) {
      result.saved_mask_paths =
          j["saved_mask_paths"].get<std::map<std::string, std::string>>();
    }
  } catch (const std::exception& e) {
    std::cerr << "Failed to parse /video/propagate_in_video response: "
              << e.what() << std::endl;
  }

  return result;
}

bool BackendInterface::videoClearAllPromptsInFrame(int frame_idx, int obj_id) {
  nlohmann::json j;
  j["frame_idx"] = frame_idx;
  j["obj_id"] = obj_id;
  std::string resp = httpPost("/video/clear_all_prompts_in_frame", j.dump());
  return !resp.empty();
}

bool BackendInterface::videoRemoveObject(int obj_id) {
  nlohmann::json j;
  j["obj_id"] = obj_id;
  std::string resp = httpPost("/video/remove_object", j.dump());
  return !resp.empty();
}

// Tracking endpoints
BackendInterface::TrackingLoadVideoResponse BackendInterface::trackingLoadVideo(
    const TrackingLoadVideoRequest& request) {
  auto body = serializeTrackingLoadVideoRequest(request);
  std::string resp = httpPost("/tracking/load_video", body);

  TrackingLoadVideoResponse result;
  if (resp.empty()) {
    return result;
  }

  try {
    auto j = nlohmann::json::parse(resp);
    result.message = j.value("message", "");
    result.shape = j.at("shape").get<std::vector<int>>();
    result.num_frames = j.at("num_frames").get<int>();
  } catch (const std::exception& e) {
    std::cerr << "Failed to parse /tracking/load_video response: " << e.what()
              << std::endl;
  }

  return result;
}

BackendInterface::TrackingGridResponse BackendInterface::trackingTrackGrid(
    const TrackingGridRequest& request) {
  auto body = serializeTrackingGridRequest(request);
  std::string resp = httpPost("/tracking/track_grid", body);

  TrackingGridResponse result;
  if (resp.empty()) {
    return result;
  }

  try {
    auto j = nlohmann::json::parse(resp);
    result.message = j.value("message", "");
    result.tracks =
        j.at("tracks").get<std::vector<std::vector<std::vector<float>>>>();
    result.visibility =
        j.at("visibility").get<std::vector<std::vector<bool>>>();
    result.num_points = j.at("num_points").get<int>();
    result.num_frames = j.at("num_frames").get<int>();
    result.output_video_path = j.value("output_video_path", "");
  } catch (const std::exception& e) {
    std::cerr << "Failed to parse /tracking/track_grid response: " << e.what()
              << std::endl;
  }

  return result;
}

BackendInterface::TrackingPointsResponse BackendInterface::trackingTrackPoints(
    const TrackingPointsRequest& request) {
  auto body = serializeTrackingPointsRequest(request);
  std::string resp = httpPost("/tracking/track_points", body);

  TrackingPointsResponse result;
  if (resp.empty()) {
    return result;
  }

  try {
    auto j = nlohmann::json::parse(resp);
    result.message = j.value("message", "");
    result.tracks =
        j.at("tracks").get<std::vector<std::vector<std::vector<float>>>>();
    result.visibility =
        j.at("visibility").get<std::vector<std::vector<bool>>>();
    result.num_points = j.at("num_points").get<int>();
    result.num_frames = j.at("num_frames").get<int>();
    result.output_video_path = j.value("output_video_path", "");
  } catch (const std::exception& e) {
    std::cerr << "Failed to parse /tracking/track_points response: " << e.what()
              << std::endl;
  }

  return result;
}

// HTTP helper methods
std::string BackendInterface::httpGet(const std::string& endpoint,
                                      bool logFailures) {
  CURL* curl = curl_easy_init();
  if (!curl) {
    std::cerr << "Failed to initialize CURL for GET" << std::endl;
    return "";
  }

  std::string readBuffer;
  std::string url = base_url_ + endpoint;

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(
      curl, CURLOPT_WRITEFUNCTION,
      +[](char* ptr, size_t size, size_t nmemb, void* userdata) -> size_t {
        auto* buffer = static_cast<std::string*>(userdata);
        buffer->append(ptr, size * nmemb);
        return size * nmemb;
      });
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

  CURLcode res = curl_easy_perform(curl);
  if (res != CURLE_OK) {
    if (logFailures) {
      std::cerr << "CURL GET failed: " << curl_easy_strerror(res) << std::endl;
    }
    readBuffer.clear();
  }

  curl_easy_cleanup(curl);
  return readBuffer;
}

std::string BackendInterface::httpPost(const std::string& endpoint,
                                       const std::string& json_body) {
  CURL* curl = curl_easy_init();
  if (!curl) {
    std::cerr << "Failed to initialize CURL for POST" << std::endl;
    return "";
  }

  std::string readBuffer;
  std::string url = base_url_ + endpoint;

  struct curl_slist* headers = nullptr;
  headers = curl_slist_append(headers, "Content-Type: application/json");

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_POST, 1L);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_body.c_str());
  curl_easy_setopt(
      curl, CURLOPT_WRITEFUNCTION,
      +[](char* ptr, size_t size, size_t nmemb, void* userdata) -> size_t {
        auto* buffer = static_cast<std::string*>(userdata);
        buffer->append(ptr, size * nmemb);
        return size * nmemb;
      });
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

  CURLcode res = curl_easy_perform(curl);
  if (res != CURLE_OK) {
    std::cerr << "CURL POST failed: " << curl_easy_strerror(res) << std::endl;
    readBuffer.clear();
  }

  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);
  return readBuffer;
}

// JSON conversion helpers
std::string BackendInterface::serializeVideoAddPointsOrBoxRequest(
    const VideoAddPointsOrBoxRequest& request) {
  nlohmann::json j;
  j["frame_idx"] = request.frame_idx;
  j["obj_id"] = request.obj_id;
  if (request.points.has_value()) {
    j["points"] = *request.points;
  } else {
    j["points"] = nullptr;
  }
  if (request.labels.has_value()) {
    j["labels"] = *request.labels;
  } else {
    j["labels"] = nullptr;
  }
  j["clear_old_points"] = request.clear_old_points;
  if (request.box.has_value()) {
    j["box"] = *request.box;
  } else {
    j["box"] = nullptr;
  }
  return j.dump();
}

std::string BackendInterface::serializeVideoAddMaskRequest(
    const VideoAddMaskRequest& request) {
  nlohmann::json j;
  j["frame_idx"] = request.frame_idx;
  j["obj_id"] = request.obj_id;
  j["mask"] = request.mask;
  return j.dump();
}

std::string BackendInterface::serializeVideoPropagateRequest(
    const VideoPropagateRequest& request) {
  nlohmann::json j;
  if (request.start_frame_idx.has_value()) {
    j["start_frame_idx"] = *request.start_frame_idx;
  } else {
    j["start_frame_idx"] = nullptr;
  }
  if (request.max_frame_num_to_track.has_value()) {
    j["max_frame_num_to_track"] = *request.max_frame_num_to_track;
  } else {
    j["max_frame_num_to_track"] = nullptr;
  }
  j["reverse"] = request.reverse;
  return j.dump();
}

std::string BackendInterface::serializeTrackingLoadVideoRequest(
    const TrackingLoadVideoRequest& request) {
  nlohmann::json j;
  j["video_path"] = request.video_path;
  return j.dump();
}

std::string BackendInterface::serializeTrackingGridRequest(
    const TrackingGridRequest& request) {
  nlohmann::json j;
  j["grid_size"] = request.grid_size;
  j["add_support_grid"] = request.add_support_grid;
  return j.dump();
}

std::string BackendInterface::serializeTrackingPointsRequest(
    const TrackingPointsRequest& request) {
  nlohmann::json j;
  j["queries"] = request.queries;
  j["add_support_grid"] = request.add_support_grid;
  return j.dump();
}
