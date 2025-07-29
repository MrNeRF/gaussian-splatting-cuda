#include "loader/loader.hpp"
#include "loader_service.hpp"
#include <memory>

namespace gs::loader {

    namespace {
        // Implementation class that hides all internal details
        class LoaderImpl : public Loader {
        public:
            LoaderImpl() : service_(std::make_unique<LoaderService>()) {}

            std::expected<LoadResult, std::string> load(
                const std::filesystem::path& path,
                const LoadOptions& options) override {

                // Just delegate to the service
                return service_->load(path, options);
            }

            bool canLoad(const std::filesystem::path& path) const override {
                // Check if any registered loader can handle this path
                // We would need to expose this functionality from the service
                // For now, let's check common extensions
                if (!std::filesystem::exists(path)) {
                    return false;
                }

                auto ext = path.extension().string();
                auto extensions = service_->getSupportedExtensions();
                return std::find(extensions.begin(), extensions.end(), ext) != extensions.end();
            }

            std::vector<std::string> getSupportedFormats() const override {
                return service_->getAvailableLoaders();
            }

            std::vector<std::string> getSupportedExtensions() const override {
                return service_->getSupportedExtensions();
            }

        private:
            std::unique_ptr<LoaderService> service_;
        };
    } // namespace

    // Factory method implementation
    std::unique_ptr<Loader> Loader::create() {
        return std::make_unique<LoaderImpl>();
    }

} // namespace gs::loader
