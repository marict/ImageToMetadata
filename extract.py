import torch

from invokeai.app.invocations.primitives import ImageField 

from .baseinvocation import (BaseInvocation, BaseInvocationOutput,
                             FieldDescriptions, InputField,
                             InvocationContext, OutputField,
                             invocation, invocation_output)

@invocation_output("metadata_outputs")
class MetaDataOutputs(BaseInvocationOutput):
    positive_prompt: str = OutputField(description="positive prompt")
    negative_prompt: str = OutputField(description="negative prompt")
    width: int = OutputField(description=FieldDescriptions.width)
    height: int = OutputField(description=FieldDescriptions.height)


@invocation(
    "i2m",
    title="Image to Metadata",
    tags=["latents", "image", "vae", "i2l"],
    category="image",
    version="1.0.0",
)
class ImageToMetadata(BaseInvocation):
    """Extracts information from an Image's Metadata"""

    image: ImageField = InputField(
        description="The image to extract prompt from",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> MetaDataOutputs:
        metadata = context.services.images.get_metadata(self.image.image_name)
        if metadata.metadata is None:
            raise ValueError(f"Metadata for image {self.image.image_name} is None")
        positive_prompt = metadata.metadata["positive_prompt"]
        negative_prompt = metadata.metadata["negative_prompt"]
        width = metadata.metadata["width"]
        height = metadata.metadata["height"]
        return MetaDataOutputs(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
        )
