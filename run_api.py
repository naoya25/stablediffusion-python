import requests
import os
import json
import datetime
from enum import Enum
from dotenv import load_dotenv

load_dotenv()


class ASPECT(Enum):
    R16_9 = "16:9"
    R1_1 = "1:1"
    R21_9 = "21:9"
    R2_3 = "2:3"
    R3_2 = "3:2"
    R4_5 = "4:5"
    R5_4 = "5:4"
    R9_16 = "9:16"
    R9_21 = "9:21"


class ImageExtension(Enum):
    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"


class PromptData:
    def __init__(
        self,
        prompt: str,
        negative_prompt: str = "",
        aspect_ratio: ASPECT = ASPECT.R1_1,
        seed: int = 0,
        output_format: ImageExtension = ImageExtension.PNG,
    ):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.aspect_ratio = aspect_ratio.value
        self.seed = seed
        self.output_format = output_format.value

    def to_dict(self):
        return {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "aspect_ratio": self.aspect_ratio,
            "seed": self.seed,
            "output_format": self.output_format,
        }


def generate_image(data: PromptData):
    # Stability API経由で画像を生成
    response = requests.post(
        f"https://api.stability.ai/v2beta/stable-image/generate/sd3",
        headers={
            "authorization": f"Bearer {os.getenv('STABILITY_API_KEY')}",
            "accept": "image/*",
        },
        files={"none": ""},
        data=data.to_dict(),
    )

    if response.status_code == 200:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("./outputs", timestamp)

        os.makedirs(output_dir, exist_ok=True)

        # 画像を保存
        image_path = os.path.join(output_dir, f"image.{data.output_format}")
        with open(image_path, "wb") as file:
            file.write(response.content)

        # レスポンスヘッダー情報を保存
        headers_path = os.path.join(output_dir, "response_headers.json")
        with open(headers_path, "w", encoding="utf-8") as file:
            # dictに変換して保存
            headers_dict = dict(response.headers)
            json.dump(headers_dict, file, ensure_ascii=False, indent=4)

        # プロンプトなどのデータを保存
        data_path = os.path.join(output_dir, "data.json")
        with open(data_path, "w", encoding="utf-8") as file:
            data_dict = data.to_dict()
            data_dict["generated_at"] = timestamp
            json.dump(data_dict, file, ensure_ascii=False, indent=4)

        print(f"Generated content saved to: {output_dir}")
        return output_dir
    else:
        raise Exception(str(response.json()))


def main():
    try:
        data = PromptData(
            prompt="1 girl, in a maid outfit, in a hotel room, standing pose",
            aspect_ratio=ASPECT.R9_16,
        )
        generate_image(data)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
