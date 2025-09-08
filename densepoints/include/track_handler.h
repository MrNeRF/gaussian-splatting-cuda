#pragma once

#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <tuple>
#include <optional>
#include <torch/torch.h>

namespace densepcd {
    struct Observation {
        int camera_id;
        int feature_id;
        float u, v;
        unsigned char rgb[3];

        Observation(int cam_id, int feat_id, float u, float v, unsigned char rgb[3]) : camera_id(cam_id), feature_id(feat_id), u(u), v(v) {
            this->rgb[0] = rgb[0];
            this->rgb[1] = rgb[1];
            this->rgb[2] = rgb[2];
        }
    };

    struct Track {
        int id;
        std::vector<Observation> observations;
        std::optional<torch::Tensor> point3D; // [3]
    };

    struct PairHash {
        std::size_t operator()(const std::pair<int, int>& p) const noexcept {
            return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
        }
    };

    struct SimpleCamera {
        int id;
        torch::Tensor P; // Projection matrix [3x4]
    };

    class TrackHandler {
    public:
        TrackHandler() : nextTrackId(0) {}

        void addCamera(int camId, const torch::Tensor& P) {
            cameras[camId] = SimpleCamera{camId, P.clone()};
        }

        // Insert a match between (camA,featA) and (camB,featB)
        void insertMatch(int camA, int featA, double uA, double vA, unsigned char rgbA[3],
                         int camB, int featB, double uB, double vB, unsigned char rgbB[3]) {
            auto keyA = std::make_pair(camA, featA);
            auto keyB = std::make_pair(camB, featB);

            bool hasA = featureToTrack.count(keyA);
            bool hasB = featureToTrack.count(keyB);

            if (!hasA && !hasB) {
                int tid = newTrack();
                addObs(tid, camA, featA, uA, vA, rgbA);
                addObs(tid, camB, featB, uB, vB, rgbB);
            } else if (hasA && !hasB) {
                int tid = featureToTrack[keyA];
                addObs(tid, camB, featB, uB, vB, rgbB);
            } else if (!hasA && hasB) {
                int tid = featureToTrack[keyB];
                addObs(tid, camA, featA, uA, vA, rgbA);
            } else {
                int tidA = featureToTrack[keyA];
                int tidB = featureToTrack[keyB];
                if (tidA != tidB) {
                    mergeTracks(tidA, tidB);
                }
            }
        }

        void triangulate() {
            for (auto& [tid, track] : tracks) {
                triangulateTrack(tid);
            }
        }

        void printTracks() const {
            for (const auto& [tid, track] : tracks) {
                std::cout << "Track " << tid << " with " << track.observations.size() << " obs:\n";
                for (const auto& obs : track.observations) {
                    std::cout << "   Cam " << obs.camera_id
                              << " Feat " << obs.feature_id
                              << " (" << obs.u << "," << obs.v << ")\n";
                }
                if (track.point3D.has_value()) {
                    auto X = track.point3D.value();
                    std::cout << "   3D: ("
                              << X[0].item<double>() << ", "
                              << X[1].item<double>() << ", "
                              << X[2].item<double>() << ")\n";
                }
            }
        }

        std::pair<torch::Tensor, torch::Tensor> get_point_cloud();

        int numCameras() const { return cameras.size(); }

    private:
        int nextTrackId;
        std::unordered_map<int, Track> tracks;
        std::unordered_map<std::pair<int, int>, int, PairHash> featureToTrack;
        std::unordered_map<int, SimpleCamera> cameras;

        int newTrack() {
            int tid = nextTrackId++;
            tracks[tid] = Track{tid, {}, std::nullopt};
            return tid;
        }

        void addObs(int tid, int cam, int feat, float u, float v, unsigned char rgb[3]) {
            auto key = std::make_pair(cam, feat);
            if (featureToTrack.count(key))
                return;

            tracks[tid].observations.emplace_back(cam, feat, u, v, rgb);
            featureToTrack[key] = tid;
        }

        void mergeTracks(int tidA, int tidB) {
            if (!tracks.count(tidA) || !tracks.count(tidB))
                return;
            if (tidA == tidB)
                return;

            for (auto& obs : tracks[tidB].observations) {
                auto key = std::make_pair(obs.camera_id, obs.feature_id);
                featureToTrack[key] = tidA;
                tracks[tidA].observations.push_back(obs);
            }
            tracks.erase(tidB);
        }

        void triangulateTrack(int tid);
    };
} // namespace densepcd