import csv
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
    get_output_dir,
)


def get_images(folder):

    paths = []

    for extension in [
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.bmp"
    ]:

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

    path = get_calibration_file(
        camera_name
    )

    if not os.path.exists(path):
        return None, None

    data = np.load(
        path,
        allow_pickle=True
    )

    return (
        data["K"],
        data["dist"]
    )


def estimate_pose_from_image(
    img,
    K,
    dist
):

    pattern_size = (
        BOARD_COLS,
        BOARD_ROWS
    )

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    found, corners = cv2.findChessboardCorners(
        gray,
        pattern_size
    )

    if not found:

        raise RuntimeError(
            "Tabuleiro não encontrado"
        )

    corners = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        CORNER_SUBPIX_CRITERIA
    )

    objp = np.zeros(
        (
            BOARD_ROWS * BOARD_COLS,
            3
        ),
        np.float32
    )

    objp[:, :2] = np.mgrid[
        0:BOARD_COLS,
        0:BOARD_ROWS
    ].T.reshape(-1, 2)

    objp *= SQUARE_SIZE_MM

    ok, rvec, tvec = cv2.solvePnP(
        objp,
        corners,
        K,
        dist
    )

    if not ok:

        raise RuntimeError(
            "solvePnP falhou"
        )

    return (
        rvec,
        tvec,
        corners,
        objp
    )


def project_and_draw(
    img,
    points_3d,
    rvec,
    tvec,
    K,
    dist,
    labels
):

    points_2d, _ = cv2.projectPoints(
        np.array(
            points_3d,
            dtype=np.float32
        ),
        rvec,
        tvec,
        K,
        dist
    )

    points_2d = points_2d.reshape(
        -1,
        2
    )

    output = img.copy()

    for (
        label,
        point
    ) in zip(
        labels,
        points_2d
    ):

        u, v = point

        u = int(round(u))
        v = int(round(v))

        cv2.circle(
            output,
            (u, v),
            7,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            output,
            label,
            (u + 10, v - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

    return (
        output,
        points_2d
    )


def main():

    largura = (
        BOARD_COLS - 1
    ) * SQUARE_SIZE_MM

    altura = (
        BOARD_ROWS - 1
    ) * SQUARE_SIZE_MM

    points_3d = [

        (0, 0, 0),

        (
            largura,
            0,
            0
        ),

        (
            0,
            altura,
            0
        ),

        (
            largura,
            altura,
            0
        ),

        (
            largura / 2,
            altura / 2,
            -50
        ),
    ]

    labels = [
        "origem",
        "canto_dir",
        "canto_baixo",
        "canto_dir_baixo",
        "ponto_alto",
    ]

    cameras = get_camera_dirs()

    for camera_name, camera_dir in cameras:

        print("\n" + "=" * 70)

        print(
            f"PROJEÇÃO 3D -> 2D: {camera_name}"
        )

        print("=" * 70)

        K, dist = load_calibration(
            camera_name
        )

        if K is None:

            print(
                "Calibração não encontrada."
            )

            continue

        images = get_images(
            camera_dir
        )

        output_root = get_output_dir(
            camera_name
        )

        output_dir = os.path.join(
            output_root,
            "projecoes"
        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        csv_path = os.path.join(
            output_dir,
            "coordenadas.csv"
        )

        rows = []

        for img_path in images:

            filename = os.path.basename(
                img_path
            )

            name, _ = os.path.splitext(
                filename
            )

            img = cv2.imread(
                img_path
            )

            if img is None:
                continue

            try:

                (
                    rvec,
                    tvec,
                    corners,
                    objp
                ) = estimate_pose_from_image(
                    img,
                    K,
                    dist
                )

            except RuntimeError:

                print(
                    f"[IGNORADA] {filename}"
                )

                continue

            output, points_2d = project_and_draw(
                img,
                points_3d,
                rvec,
                tvec,
                K,
                dist,
                labels
            )

            # Erro de reprojeção
            projected_corners, _ = cv2.projectPoints(
                objp,
                rvec,
                tvec,
                K,
                dist
            )

            projected_corners = projected_corners.reshape(
                -1,
                2
            )

            detected = corners.reshape(
                -1,
                2
            )

            errors = np.linalg.norm(
                projected_corners - detected,
                axis=1
            )

            mean_error = errors.mean()

            for (
                label,
                p3,
                p2
            ) in zip(
                labels,
                points_3d,
                points_2d
            ):

                rows.append(
                    [
                        camera_name,
                        filename,
                        label,

                        p3[0],
                        p3[1],
                        p3[2],

                        p2[0],
                        p2[1],

                        mean_error,
                    ]
                )

            output_path = os.path.join(
                output_dir,
                f"{name}_projecao.png"
            )

            cv2.imwrite(
                output_path,
                output
            )

            print(
                f"[OK] {filename} "
                f"| erro médio = {mean_error:.3f}px"
            )

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(
                file,
                delimiter=";"
            )

            writer.writerow(
                [
                    "camera",
                    "imagem",
                    "ponto",
                    "X_mm",
                    "Y_mm",
                    "Z_mm",
                    "u_px",
                    "v_px",
                    "erro_reprojecao_px",
                ]
            )

            writer.writerows(
                rows
            )

        print(
            f"\nCSV salvo em: {csv_path}"
        )


if __name__ == "__main__":
    main()
