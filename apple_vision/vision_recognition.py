import Quartz
import Vision
from Cocoa import NSURL
from Foundation import NSDictionary
from wurlitzer import pipes


def image_to_text(img_path, lang="ru"):
    input_url = NSURL.fileURLWithPath_(img_path)

    with pipes() as (out, err):
        input_image = Quartz.CIImage.imageWithContentsOfURL_(input_url)

    vision_options = NSDictionary.dictionaryWithDictionary_({})
    vision_handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(
        input_image, vision_options
    )
    results = []
    handler = make_request_handler(results)
    vision_request = Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(handler)

    # Настройка для распознавания текста
    vision_request.recognitionLevel = Vision.VNRequestTextRecognitionLevelAccurate
    vision_request.usesLanguageCorrection = True

    # Устанавливаем язык распознавания текста
    vision_request.recognitionLanguages = [lang]

    error = vision_handler.performRequests_error_([vision_request], None)

    return results


def make_request_handler(results):
    if not isinstance(results, list):
        raise ValueError("results must be a list")

    def handler(request, error):
        if error:
            print(f"Error! {error}")
        else:
            observations = request.results()
            for text_observation in observations:
                recognized_text = text_observation.topCandidates_(1)[0]
                results.append([recognized_text.string(), recognized_text.confidence()])

    return handler
