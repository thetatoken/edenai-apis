"""
Theta EdgeCloud On-Demand Models API Provider

Provides access to AI inference models running on Theta EdgeCloud's
decentralized GPU network.

To see all available services, use the list services API:
GET https://ondemand.thetaedgecloud.com/service/list

Documentation: https://docs.thetaedgecloud.com
API Base URL: https://ondemand.thetaedgecloud.com
"""

import base64
import time
import uuid
from io import BytesIO
from typing import Any, Dict, List, Literal, Optional

import requests

from edenai_apis.features import AudioInterface, ImageInterface, ProviderInterface
from edenai_apis.features.audio.speech_to_text_async.speech_to_text_async_dataclass import (
    SpeechDiarization,
    SpeechToTextAsyncDataClass,
)
from edenai_apis.features.image.generation import (
    GeneratedImageDataClass,
    GenerationDataClass as ImageGenerationDataClass,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
    ResponseType,
)


class ThetaedgecloudApi(ProviderInterface, AudioInterface, ImageInterface):
    """
    Theta EdgeCloud On-Demand Models API

    Provides access to AI inference models running on Theta EdgeCloud's
    decentralized GPU network.

    https://www.thetaedgecloud.com
    """

    provider_name = "thetaedgecloud"

    def __init__(self, api_keys: Optional[Dict[str, Any]] = None) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        # On-Demand API Access Token from Theta EdgeCloud dashboard
        self.access_token = self.api_settings["access_token"]
        self.base_url = "https://ondemand.thetaedgecloud.com"
        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

    def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict] = None,
        timeout: int = 120,
    ) -> Dict:
        """Make HTTP request to Theta EdgeCloud API."""
        url = f"{self.base_url}{endpoint}"

        try:
            response = requests.request(
                method=method,
                url=url,
                headers=self.headers,
                json=json_data,
                timeout=timeout,
            )
        except requests.exceptions.Timeout:
            raise ProviderException("Request timed out", code=408)
        except requests.exceptions.RequestException as e:
            raise ProviderException(f"Request failed: {str(e)}", code=500)

        if response.status_code != 200:
            try:
                error_data = response.json()
                error_message = error_data.get("message", response.text)
            except Exception:
                error_message = response.text
            raise ProviderException(error_message, code=response.status_code)

        try:
            return response.json()
        except requests.JSONDecodeError as e:
            raise ProviderException("Invalid JSON response", code=500) from e

    def _poll_for_result(
        self, request_id: str, max_wait: int = 60, poll_interval: int = 2
    ) -> Dict:
        """Poll for async request completion."""
        start_time = time.time()

        while time.time() - start_time < max_wait:
            result = self._make_request("GET", f"/infer_request/{request_id}")
            body = result.get("body", {})
            infer_requests = body.get("infer_requests", [])

            if infer_requests:
                request = infer_requests[0]
                state = request.get("state")

                if state == "success":
                    return request
                elif state == "error":
                    error_msg = request.get("error_message", "Unknown error")
                    raise ProviderException(error_msg, code=500)
                # Still processing, continue polling

            time.sleep(poll_interval)

        raise ProviderException("Request timed out waiting for completion", code=408)

    # -------------------------------------------------------------------------
    # Audio: Speech-to-Text (Whisper)
    # -------------------------------------------------------------------------

    def audio__speech_to_text_async__launch_job(
        self,
        file: str,
        language: str,
        speakers: int,
        profanity_filter: bool,
        vocabulary: Optional[List[str]],
        audio_attributes: tuple,
        model: Optional[str] = None,
        file_url: str = "",
        provider_params: Optional[dict] = None,
        **kwargs,
    ) -> AsyncLaunchJobResponseType:
        """
        Launch async speech-to-text job using Theta EdgeCloud Whisper.

        Args:
            file: Local file path to audio file
            language: Language code (e.g., 'en', 'es', 'fr')
            speakers: Number of speakers (not used by Whisper)
            profanity_filter: Filter profanity (not used by Whisper)
            vocabulary: Custom vocabulary (not used by Whisper)
            audio_attributes: Audio metadata tuple
            model: Model variant (optional)
            file_url: URL to audio file (alternative to local file)
            provider_params: Additional provider-specific parameters

        Returns:
            AsyncLaunchJobResponseType with provider_job_id
        """
        provider_params = provider_params or {}

        # Determine audio source - prefer URL if provided
        if file_url:
            audio_input = file_url
        elif file:
            # For local files, we need to use presigned URL upload
            # For now, raise an error - URL input is required
            raise ProviderException(
                "Local file upload not yet supported. Please provide file_url.",
                code=400,
            )
        else:
            raise ProviderException("No audio file or URL provided", code=400)

        # Build request payload
        payload = {
            "input": {
                "audio_filename": audio_input,
            },
            "wait": 0,  # Async mode - return immediately
        }

        # Add language if specified
        if language:
            payload["input"]["language"] = language

        # Launch the job
        result = self._make_request("POST", "/infer_request/whisper", json_data=payload)

        body = result.get("body", {})
        infer_requests = body.get("infer_requests", [])

        if not infer_requests:
            raise ProviderException("No inference request created", code=500)

        request_id = infer_requests[0].get("id")
        if not request_id:
            raise ProviderException("No request ID returned", code=500)

        return AsyncLaunchJobResponseType(provider_job_id=request_id)

    def audio__speech_to_text_async__get_job_result(
        self,
        provider_job_id: str,
        **kwargs,
    ) -> AsyncBaseResponseType[SpeechToTextAsyncDataClass]:
        """
        Get result of async speech-to-text job.

        Args:
            provider_job_id: The inference request ID from launch_job

        Returns:
            AsyncResponseType with transcribed text or AsyncPendingResponseType
        """
        result = self._make_request("GET", f"/infer_request/{provider_job_id}")

        body = result.get("body", {})
        infer_requests = body.get("infer_requests", [])

        if not infer_requests:
            raise ProviderException("Request not found", code=404)

        request = infer_requests[0]
        state = request.get("state")

        if state == "success":
            output = request.get("output", {})
            text = output.get("text", "")

            # Build standardized response
            diarization = SpeechDiarization(total_speakers=0, entries=[])
            standardized_response = SpeechToTextAsyncDataClass(
                text=text, diarization=diarization
            )

            return AsyncResponseType[SpeechToTextAsyncDataClass](
                original_response=request,
                standardized_response=standardized_response,
                provider_job_id=provider_job_id,
            )

        elif state == "error":
            error_msg = request.get("error_message", "Unknown error")
            raise ProviderException(error_msg, code=500)

        else:
            # Still processing
            return AsyncPendingResponseType[SpeechToTextAsyncDataClass](
                provider_job_id=provider_job_id
            )

    # -------------------------------------------------------------------------
    # Image: Generation (FLUX)
    # -------------------------------------------------------------------------

    def image__generation(
        self,
        text: str,
        resolution: Literal["256x256", "512x512", "1024x1024"],
        num_images: int = 1,
        model: Optional[str] = None,
        **kwargs,
    ) -> ResponseType[ImageGenerationDataClass]:
        """
        Generate images using Theta EdgeCloud FLUX.

        Args:
            text: The prompt describing the image to generate
            resolution: Output resolution (1024x1024 recommended for FLUX)
            num_images: Number of images to generate (currently 1)
            model: Model variant (optional)

        Returns:
            ResponseType with generated image data
        """
        # Parse resolution
        try:
            width, height = map(int, resolution.split("x"))
        except ValueError:
            width, height = 1024, 1024

        # Build request payload
        payload = {
            "input": {
                "prompt": text,
                "width": width,
                "height": height,
            },
            "wait": 60,  # Wait for completion (sync mode)
        }

        # Launch and wait for result
        result = self._make_request(
            "POST", "/infer_request/flux", json_data=payload, timeout=120
        )

        body = result.get("body", {})
        infer_requests = body.get("infer_requests", [])

        if not infer_requests:
            raise ProviderException("No inference request created", code=500)

        request = infer_requests[0]
        state = request.get("state")

        if state != "success":
            # If not complete, poll for result
            request_id = request.get("id")
            if request_id:
                request = self._poll_for_result(request_id)
            else:
                raise ProviderException("Image generation failed", code=500)

        output = request.get("output", {})
        image_url = output.get("image_url", "")

        if not image_url:
            raise ProviderException("No image URL in response", code=500)

        # Build standardized response
        # Note: Eden AI expects base64 image data OR image_resource_url
        # We'll provide the URL directly
        generated_images = [
            GeneratedImageDataClass(
                image="",  # Base64 not provided, using URL instead
                image_resource_url=image_url,
            )
        ]

        standardized_response = ImageGenerationDataClass(items=generated_images)

        return ResponseType[ImageGenerationDataClass](
            original_response=request,
            standardized_response=standardized_response,
        )