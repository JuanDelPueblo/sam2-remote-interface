import torch
import numpy as np
import os
from sam2.sam2_video_predictor import SAM2VideoPredictor

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class SAM2VideoMasker:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"Utilizing device: {self.device}")

        if self.device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # enable tf32 for new GPUs
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif self.device.type == "mps":
            print(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS."
            )

        self.predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")

        self.inference_state = None
    
    def init_state(self, video_dir):
        self.inference_state = self.predictor.init_state(video_path=video_dir)
        self.predictor.reset_state(self.inference_state)

    def reset_state(self):
        self.predictor.reset_state(self.inference_state)

    def add_new_points_or_box(self, frame_idx, obj_id, points=None, labels=None, clear_old_points=True, box=None):
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
            clear_old_points=clear_old_points,
            box=box,
        )

        return out_obj_ids, out_mask_logits

    def propagate_in_video(self, start_frame_idx=None, max_frame_num_to_track=None, reverse=False):
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state, start_frame_idx, max_frame_num_to_track, reverse):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        return video_segments

    def clear_all_prompts_in_frame(self, frame_idx, obj_id):
        self.predictor.clear_all_prompts_in_frame(
            inference_state=self.inference_state,
            frame_idx=frame_idx, obj_id=obj_id
        )
        return

    def remove_object(self, obj_id):
        self.predictor.remove_object(
            inference_state=self.inference_state,
            obj_id=obj_id
        )