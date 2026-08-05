def blockwise_causal_attention_mask(
    q_idx,
    kv_idx,
    ends,
    frame_seqlen: int,
    local_attn_size: int = -1,
    sink_size: int = 0,
):
    """Mask predicate shared by Stage2/Stage3 blockwise FlexAttention."""
    causal = kv_idx < ends[q_idx]
    if local_attn_size == -1:
        visible = causal
    else:
        sink_end = sink_size * frame_seqlen
        recent_size = (local_attn_size - sink_size) * frame_seqlen
        sink = kv_idx < sink_end
        recent = kv_idx >= ends[q_idx] - recent_size
        visible = causal & (sink | recent)

    # Keep padded query rows safe for FlexAttention.
    return visible | (q_idx == kv_idx)
