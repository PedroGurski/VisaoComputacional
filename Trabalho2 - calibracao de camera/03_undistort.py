"""
03_undistort.py

Remove a distorção de TODAS as imagens presentes em:

data/imgs/<camera>/
"""

import glob
import os

import cv2
import numpy as np

from config import (
    get_camera_dirs,
    get_calibration_file,
    get_output_dir,
)


def get_images(folder):

    extensions = [
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.bmp"
    ]

    paths = []

    for extension in extensions:

        paths.extend(
            glob.glob(
                os.path.join(
                    folder,
                    extension
                )
            )
        )

    return sorted(paths)


def load_calibration(camera_name):

    calib_file = get_calibration_file(
        camera_name
    )

    if not os.path.exists(calib_file):

        print(
            f"Calibração não encontrada: {calib_file}"
        )

        return None, None

    data = np.load(
        calib_file,
        allow_pickle=True
    )

    return (
        data["K"],
        data["dist"]
    )


def undistort_image(img, K, dist):

    h, w = img.shape[:2]

    new_K, roi = cv2.getOptimalNewCameraMatrix(
        K,
        dist,
        (w, h),
        alpha=1,
        newImgSize=(w, h)
    )

    corrected = cv2.undistort(
        img,
        K,
        dist,
        None,
        new_K
    )

    x, y, rw, rh = roi

    if rw > 0 and rh > 0:

        cropped = corrected[
            y:y + rh,
            x:x + rw
        ]

    else:

        cropped = corrected

    return (
        corrected,
        cropped
    )


def main():

    cameras = get_camera_dirs()

    for camera_name, camera_dir in cameras:

        print("\n" + "=" * 70)

        print(
            f"CORRIGINDO CÂMERA: {camera_name}"
        )

        print("=" * 70)

        K, dist = load_calibration(
            camera_name
        )

        if K is None:
            continue

        images = get_images(
            camera_dir
        )

        output_root = get_output_dir(
            camera_name
        )

        corrected_dir = os.path.join(
            output_root,
            "corrigidas"
        )

        cropped_dir = os.path.join(
            output_root,
            "recortadas"
        )

        comparison_dir = os.path.join(
            output_root,
            "comparacoes"
        )

        os.makedirs(
            corrected_dir,
            exist_ok=True
        )

        os.makedirs(
            cropped_dir,
            exist_ok=True
        )

        os.makedirs(
            comparison_dir,
            exist_ok=True
        )

        for img_path in images:

            filename = os.path.basename(
                img_path
            )

            name, ext = os.path.splitext(
                filename
            )

            img = cv2.imread(
                img_path
            )

            if img is None:

                print(
                    f"[ERRO] {filename}"
                )

                continue

            corrected, cropped = undistort_image(
                img,
                K,
                dist
            )

            comparison = np.hstack(
                [
                    img,
                    corrected
                ]
            )

            cv2.imwrite(
                os.path.join(
                    corrected_dir,
                    f"{name}_corrigida.png"
                ),
                corrected
            )

            cv2.imwrite(
                os.path.join(
                    cropped_dir,
                    f"{name}_recortada.png"
                ),
                cropped
            )

            cv2.imwrite(
                os.path.join(
                    comparison_dir,
                    f"{name}_comparacao.png"
                ),
                comparison
            )

            print(
                f"[OK] {filename}"
            )

    print(
        "\nTodas as imagens foram corrigidas."
    )


if __name__ == "__main__":
    main()