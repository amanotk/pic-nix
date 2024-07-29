// -*- C++ -*-
#ifndef _RESOURCE_DIAG_HPP_
#define _RESOURCE_DIAG_HPP_

#include "base.hpp"

///
/// @brief Diagnostic for resource usage
///
template <typename App, typename Data>
class ResourceDiag : public BaseDiag<App, Data>
{
protected:
  using BaseDiag<App, Data>::info;

  // calculate statistics
  template <typename T>
  auto statistics(T& data)
  {
    // sort
    std::sort(data.begin(), data.end());

    json stat      = {};
    stat["min"]    = data.front();
    stat["max"]    = data.back();
    stat["mean"]   = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    stat["quant1"] = this->percentile(data, 0.25, true);
    stat["quant2"] = this->percentile(data, 0.50, true);
    stat["quant3"] = this->percentile(data, 0.75, true);
    stat["size"]   = data.size();

    return stat;
  }

public:
  // constructor
  ResourceDiag(std::shared_ptr<DiagInfo> info) : BaseDiag<App, Data>("resource", info)
  {
  }

  void operator()(json& config, App& app, Data& data) override
  {
    if (this->require_diagnostic(data.curstep, config) == false)
      return;

    const float64        to_gb        = 1.0 / (1024 * 1024 * 1024);
    int                  local_chunk  = 0;
    float64              local_memory = 0;
    float64              local_load   = 0;
    float64              total_load   = 0;
    std::vector<float64> memoryvec;
    std::vector<float64> loadvec;
    std::vector<int>     node_chunk;
    std::vector<float64> node_memory;
    std::vector<float64> node_load;
    std::vector<int>     rank_chunk;
    std::vector<float64> rank_memory;
    std::vector<float64> rank_load;

    //
    // local resource usage
    //
    local_chunk = data.chunkvec.size();
    memoryvec.resize(local_chunk);
    loadvec.resize(local_chunk);
    for (int i = 0; i < local_chunk; i++) {
      memoryvec[i] = data.chunkvec[i]->get_size_byte() * to_gb;
      loadvec[i]   = data.chunkvec[i]->get_total_load();
    }
    local_memory = std::accumulate(memoryvec.begin(), memoryvec.end(), 0.0);
    local_load   = std::accumulate(loadvec.begin(), loadvec.end(), 0.0);

    // total load
    MPI_Reduce(&local_load, &total_load, 1, MPI_FLOAT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    //
    // resource usage for each node
    //
    if (config.contains("node") == true) {
      // chunk
      {
        int sum = 0;
        MPI_Reduce(&local_chunk, &sum, 1, MPI_INT, MPI_SUM, 0, info->intra_comm);

        if (info->intra_rank == 0) {
          node_chunk.resize(info->inter_size, 0);
          MPI_Gather(&sum, 1, MPI_INT, node_chunk.data(), 1, MPI_INT, 0, info->inter_comm);
        }
      }

      // memory
      {
        float64 sum = 0;
        MPI_Reduce(&local_memory, &sum, 1, MPI_FLOAT64_T, MPI_SUM, 0, info->intra_comm);

        if (info->intra_rank == 0) {
          node_memory.resize(info->inter_size, 0);
          MPI_Gather(&sum, 1, MPI_FLOAT64_T, node_memory.data(), 1, MPI_FLOAT64_T, 0,
                     info->inter_comm);
        }
      }

      // load
      {
        float64 sum = 0;
        MPI_Reduce(&local_load, &sum, 1, MPI_FLOAT64_T, MPI_SUM, 0, info->intra_comm);

        if (info->intra_rank == 0) {
          node_load.resize(info->inter_size, 0);
          MPI_Gather(&sum, 1, MPI_FLOAT64_T, node_load.data(), 1, MPI_FLOAT64_T, 0,
                     info->inter_comm);
          // normalize
          std::for_each(node_load.begin(), node_load.end(), [=](auto& x) { x /= total_load; });
        }
      }
    }

    //
    // resource usage for each rank
    //
    if (config.contains("rank") == true) {
      // chunk
      rank_chunk.resize(data.nprocess, 0);
      MPI_Gather(&local_chunk, 1, MPI_INT, rank_chunk.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

      // memory
      rank_memory.resize(data.nprocess, 0);
      MPI_Gather(&local_memory, 1, MPI_FLOAT64_T, rank_memory.data(), 1, MPI_FLOAT64_T, 0,
                 MPI_COMM_WORLD);

      // load
      rank_load.resize(data.nprocess, 0);
      MPI_Gather(&local_load, 1, MPI_FLOAT64_T, rank_load.data(), 1, MPI_FLOAT64_T, 0,
                 MPI_COMM_WORLD);
      // normalize
      std::for_each(rank_load.begin(), rank_load.end(), [=](auto& x) { x /= total_load; });
    }

    // save to file
    {
      json result = {
          {"step", data.curstep},     {"rank", data.thisrank},      {"time", data.curtime},
          {"node_chunk", node_chunk}, {"node_memory", node_memory}, {"node_load", node_load},
          {"rank_chunk", rank_chunk}, {"rank_memory", rank_memory}, {"rank_load", rank_load}};

      if (data.thisrank == 0) {
        savefile(config, result);
      }
    }
  }

  void savefile(json& config, json& result)
  {
    std::string dirname  = this->format_dirname("");
    std::string filename = dirname + "resource.msgpack";

    json record    = {};
    record["step"] = result["step"];
    record["time"] = result["time"];

    // node
    if (config.contains("node") == true) {
      auto node_chunk  = result["node_chunk"].get<std::vector<int>>();
      auto node_memory = result["node_memory"].get<std::vector<float64>>();
      auto node_load   = result["node_load"].get<std::vector<float64>>();
      json chunk       = {};
      json memory      = {};
      json load        = {};

      if (config["node"] == "stats" || config["node"] == "full") {
        chunk["stats"]  = statistics(node_chunk);
        memory["stats"] = statistics(node_memory);
        load["stats"]   = statistics(node_load);
      }

      if (config["node"] == "full") {
        chunk["full"]  = node_chunk;
        memory["full"] = node_memory;
        load["full"]   = node_load;
      }

      record["node"] = {{"chunk", chunk}, {"memory", memory}, {"load", load}};
    }

    // rank
    if (config.contains("rank") == true) {
      auto rank_chunk  = result["rank_chunk"].get<std::vector<int>>();
      auto rank_memory = result["rank_memory"].get<std::vector<float64>>();
      auto rank_load   = result["rank_load"].get<std::vector<float64>>();
      json chunk       = {};
      json memory      = {};
      json load        = {};

      if (config["rank"] == "stats" || config["rank"] == "full") {
        chunk["stats"]  = statistics(rank_chunk);
        memory["stats"] = statistics(rank_memory);
        load["stats"]   = statistics(rank_load);
      }

      if (config["rank"] == "full") {
        chunk["full"]  = rank_chunk;
        memory["full"] = rank_memory;
        load["full"]   = rank_load;
      }

      record["rank"] = {{"chunk", chunk}, {"memory", memory}, {"load", load}};
    }

    // initial call
    if (this->is_initial_step(result["step"], config) == true) {
      std::filesystem::remove(filename);
    }

    // append to file
    if (this->make_sure_directory_exists(filename) == true) {
      std::ofstream             ofs(filename, nix::binary_append);
      std::vector<std::uint8_t> buffer = json::to_msgpack(record);
      ofs.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
      ofs.close();
    }
  }
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
