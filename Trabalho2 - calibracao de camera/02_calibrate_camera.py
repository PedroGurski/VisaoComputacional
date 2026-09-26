import glob
import os
import csv

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


def build_object_points():
    objp = np.zeros((BOARD_ROWS * BOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM
    return objp


def get_images(folder):
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    paths = []

    for extension in extensions:
        paths.extend(glob.glob(os.path.join(folder, extension)))

    return sorted(paths)


def draw_colored_chessboard(img, corners, board_cols, board_rows):
    output = img.copy()
    points = corners.reshape(-1, 2)

    colors = [
        (0, 0, 255),
        (0, 128, 255),
        (0, 220, 220),
        (0, 255, 0),
        (255, 220, 0),
        (255, 0, 0),
        (255, 0, 255),
        (0, 0, 255),
        (0, 128, 255),
        (0, 220, 220),
        (0, 255, 0),
    ]

    for col in range(board_cols):
        color = colors[col % len(colors)]

        for row in range(board_rows - 1):
            idx1 = row * board_cols + col
            idx2 = (row + 1) * board_cols + col

            p1 = tuple(np.round(points[idx1]).astype(int))
            p2 = tuple(np.round(points[idx2]).astype(int))

            cv2.line(output, p1, p2, color, 2, cv2.LINE_AA)

    for index, point in enumerate(points):
        x, y = np.round(point).astype(int)
        col = index % board_cols
        color = colors[col % len(colors)]

        cv2.circle(output, (x, y), 6, color, 2, cv2.LINE_AA)
        cv2.circle(output, (x, y), 2, color, -1, cv2.LINE_AA)

        cv2.drawMarker(
            output,
            (x, y),
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=5,
            thickness=1,
            line_type=cv2.LINE_AA,
        )

    return output


def save_coordinates_csv(csv_path, rows):
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(
            [
                "imagem",
                "indice",
                "linha",
                "coluna",
                "x_pixel",
                "y_pixel",
            ]
        )
        writer.writerows(rows)


def calibrate(
    image_paths,
    pattern_size,
    corners_output_dir,
    csv_path,
    verbose=True,
):
    objp = build_object_points()

    object_points = []
    image_points = []
    used_paths = []
    img_shape = None
    csv_rows = []

    os.makedirs(corners_output_dir, exist_ok=True)

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    for path in image_paths:
        img = cv2.imread(path)

        if img is None:
            print(f"[ERRO] Não foi possível abrir: {path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(
            gray,
            pattern_size,
            flags,
        )

        if not found:
            if verbose:
                print(
                    f"[IGNORADA] Tabuleiro não encontrado: "
                    f"{os.path.basename(path)}"
                )
            continue

        corners_refined = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            CORNER_SUBPIX_CRITERIA,
        )

        object_points.append(objp.copy())
        image_points.append(corners_refined)
        used_paths.append(path)

        filename = os.path.basename(path)
        name, _ = os.path.splitext(filename)

        visualization = draw_colored_chessboard(
            img,
            corners_refined,
            BOARD_COLS,
            BOARD_ROWS,
        )

        output_path = os.path.join(
            corners_output_dir,
            f"{name}_pontos.png",
        )

        cv2.imwrite(output_path, visualization)

        points_2d = corners_refined.reshape(-1, 2)

        for index, (x, y) in enumerate(points_2d):
            linha = index // BOARD_COLS
            coluna = index % BOARD_COLS

            csv_rows.append(
                [
                    filename,
                    index,
                    linha,
                    coluna,
                    float(x),
                    float(y),
                ]
            )

        if verbose:
            print(
                f"[OK] {filename} | "
                f"{len(points_2d)} pontos encontrados"
            )

    save_coordinates_csv(csv_path, csv_rows)

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
        flags=cv2.CALIB_FIX_K3,
    )

    errors = []

    for i in range(len(object_points)):
        projected, _ = cv2.projectPoints(
            object_points[i],
            rvecs[i],
            tvecs[i],
            K,
            dist,
        )

        detected_points = image_points[i].reshape(-1, 2).astype(np.float64)
        projected_points = projected.reshape(-1, 2).astype(np.float64)

        distances = np.linalg.norm(
            detected_points - projected_points,
            axis=1,
        )

        errors.append(float(np.mean(distances)))

    mean_error = float(np.mean(errors))

    return {
        "ret": ret,
        "K": K,
        "dist": dist,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "used_paths": used_paths,
        "image_shape": img_shape,
        "mean_reprojection_error": mean_error,
        "per_image_error": errors,
    }


def main():
    cameras = get_camera_dirs()

    if not cameras:
        print("Nenhuma câmera encontrada em data/imgs.")
        return

    pattern_size = (BOARD_COLS, BOARD_ROWS)

    print(f"Câmeras encontradas: {len(cameras)}")
    print(f"Padrão: {BOARD_COLS} x {BOARD_ROWS}")
    print(f"Total de pontos por imagem: {BOARD_COLS * BOARD_ROWS}")

    for camera_name, camera_dir in cameras:
        print(f"\nCALIBRANDO: {camera_name}")

        image_paths = get_images(camera_dir)

        print(f"Imagens encontradas: {len(image_paths)}")

        if not image_paths:
            continue

        output_root = get_output_dir(camera_name)

        corners_output_dir = os.path.join(
            output_root,
            "pontos_detectados",
        )

        os.makedirs(corners_output_dir, exist_ok=True)

        csv_path = os.path.join(
            corners_output_dir,
            "coordenadas_pontos.csv",
        )

        try:
            result = calibrate(
                image_paths,
                pattern_size,
                corners_output_dir,
                csv_path,
            )
        except RuntimeError as e:
            print(f"Falha na calibração de {camera_name}: {e}")
            continue

        print(
            f"Imagens utilizadas: "
            f"{len(result['used_paths'])}/{len(image_paths)}"
        )

        print("\nMatriz intrínseca K:")
        print(result["K"])

        print("\nCoeficientes de distorção:")
        print(result["dist"].ravel())

        print("\nErro médio de reprojeção:")
        print(f"{result['mean_reprojection_error']:.4f} px")

        calib_file = get_calibration_file(camera_name)

        np.savez(
            calib_file,
            K=result["K"],
            dist=result["dist"],
            rvecs=np.array(result["rvecs"]),
            tvecs=np.array(result["tvecs"]),
            image_shape=result["image_shape"],
            used_paths=np.array(result["used_paths"]),
            mean_reprojection_error=result["mean_reprojection_error"],
            per_image_error=np.array(result["per_image_error"]),
        )

        print(f"\nCalibração salva em: {calib_file}")
        print(f"Imagens com os pontos em: {corners_output_dir}")
        print(f"Coordenadas dos pontos em: {csv_path}")


if __name__ == "__main__":
    main()
