class BackendInterface {
 public:
  BackendInterface() = default;
  ~BackendInterface() = default;

  // Initialize the HTTP communication backend
  void initialize();
  // Shutdown the HTTP communication backend
  void shutdown();

 private:
  // Add private members as needed
};