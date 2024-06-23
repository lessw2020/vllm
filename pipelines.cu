// 4-stage Pipeline
static constexpr int NumStages = 4;
using MainloopPipeline = typename cutlass::PipelineAsync<NumStages>;
using PipelineState = typename cutlass::PipelineState<NumStages>;

// 2 producer threads and 1 consumer thread
typename MainloopPipeline::Params params;
params.producer_arv_count = 2;
params.consumer_arv_count = 1;
MainloopPipeline pipeline(shared_storage.storage, params);

// Producer threads
if (thread_idx == 0 or thread_idx == 1) {
  PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();
  for ( ; iter > 0; --iter) {
    pipeline.producer_acquire(smem_pipe_write);

    // Producer ops
    // If any memory operations are involved, then we also need
    // to guarantee that writes are completed and visible to consumer(s).

    pipeline.producer_commit(smem_pipe_write);
    ++smem_pipe_write;
  }
}
else if (thread_idx == 2) {
  PipelineState smem_pipe_read;
  for (; iter > 0; --iter) {
    pipeline.consumer_wait(smem_pipe_read);

    // Consumer ops

    pipeline.consumer_release(smem_pipe_read);
    ++smem_pipe_read;
  }
}
