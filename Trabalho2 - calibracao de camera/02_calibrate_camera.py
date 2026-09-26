import glob
import os

import cv2
import numpy as np

from config import (
    BOARD_COLS,
    BOARD_ROWS,
    SQUARE_SIZE_MM,
    CORNER_SUBPIX_CRITERIA,
    get_camera_dirs,
    get_calibration_file,
)


def build_object_points():

    objp = np.zeros(
        (BOARD_ROWS * BOARD_COLS, 3),
        np.float32
    )

    objp[:, :2] = np.mgrid[
        0:BOARD_COLS,
        0:BOARD_ROWS
    ].T.reshape(-1, 2)

    objp *= SQUARE_SIZE_MM

    return objp


def calibrate(image_paths, pattern_size, verbose=True):

    objp = build_object_points()

    object_points = []
    image_points = []

    used_paths = []

    img_shape = None

    for path in image_paths:

        img = cv2.imread(path)

        if img is None:
            print(f"[ERRO] Não foi possível abrir: {path}")
            continue

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        img_shape = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(
            gray,
            pattern_size
        )

        if not found:

            if verbose:
                print(
                    f"[IGNORADA] tabuleiro não encontrado: "
                    f"{os.path.basename(path)}"
                )

            continue

        corners_refined = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            CORNER_SUBPIX_CRITERIA
        )

        object_points.append(objp)
        image_points.append(corners_refined)

        used_paths.append(path)

        if verbose:
            print(
                f"[OK] {os.path.basename(path)}"
            )

    if len(object_points) < 5:

        raise RuntimeError(
            f"Apenas {len(object_points)} imagens válidas encontradas."
        )

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        img_shape,
        None,
        None,
        flags=cv2.CALIB_FIX_K3
    )

    # ========================================================
    # ERRO DE REPROJEÇÃO
    # ========================================================

    errors = []

    for i in range(len(object_points)):

        projected, _ = cv2.projectPoints(
            object_points[i],
            rvecs[i],
            tvecs[i],
            K,
            dist
        )

        error = (
            cv2.norm(
                image_points[i],
                projected,
                cv2.NORM_L2
            )
            / len(projected)
        )

        errors.append(error)

    mean_error = np.mean(errors)

    return {
        "K": K,
        "dist": dist,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "used_paths": used_paths,
        "image_shape": img_shape,
        "mean_reprojection_error": mean_error,
        "per_image_error": errors,
    }


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


def main():

    cameras = get_camera_dirs()

    if not cameras:

        print(
            "Nenhuma pasta de câmera encontrada em data/imgs."
        )

        return

    pattern_size = (
        BOARD_COLS,
        BOARD_ROWS
    )

    print(
        f"\nCâmeras encontradas: {len(cameras)}"
    )

    for camera_name, camera_dir in cameras:

        print("\n" + "=" * 70)
        print(
            f"CALIBRANDO: {camera_name}"
        )
        print("=" * 70)

        image_paths = get_images(
            camera_dir
        )

        print(
            f"Imagens encontradas: {len(image_paths)}"
        )

        if not image_paths:

            print(
                f"Nenhuma imagem encontrada em {camera_dir}"
            )

            continue

        try:

            result = calibrate(
                image_paths,
                pattern_size
            )

        except RuntimeError as e:

            print(
                f"Falha na calibração de {camera_name}: {e}"
            )

            continue

        print(
            f"\nImagens utilizadas: "
            f"{len(result['used_paths'])}/{len(image_paths)}"
        )

        print(
            "\nMatriz intrínseca K:"
        )

        print(
            result["K"]
        )

        print(
            "\nCoeficientes de distorção:"
        )

        print(
            result["dist"].ravel()
        )

        print(
            "\nErro médio de reprojeção:"
        )

        print(
            f"{result['mean_reprojection_error']:.4f} px"
        )

        calib_file = get_calibration_file(
            camera_name
        )

        np.savez(
            calib_file,

            K=result["K"],
            dist=result["dist"],

            rvecs=np.array(
                result["rvecs"]
            ),

            tvecs=np.array(
                result["tvecs"]
            ),

            image_shape=result["image_shape"],

            used_paths=np.array(
                result["used_paths"]
            ),

            mean_reprojection_error=
            result["mean_reprojection_error"],

            per_image_error=np.array(
                result["per_image_error"]
            ),
        )

        print(
            f"\nCalibração salva em:"
        )

        print(
            calib_file
        )

    print("\n" + "=" * 70)

    print(
        "TODAS AS CÂMERAS FORAM PROCESSADAS"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
