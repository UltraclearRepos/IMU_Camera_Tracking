import numpy as np
import pycolmap


class ColmapMappingDatabase:
    """Own COLMAP camera/image records and optional IMU pose priors."""

    def __init__(
        self,
        database_path,
        camera_matrix,
        distortion,
        image_size,
        imu_gravity_provider=None,
    ):
        self.database_path = database_path
        self.camera_matrix = camera_matrix
        self.distortion = distortion
        self.image_size = image_size
        self.imu_gravity_provider = imu_gravity_provider
        self.camera_id = None
        self.camera_sensor = None
        self.rig_id = None

    def initialize(self):
        database = pycolmap.Database.open(self.database_path)
        try:
            camera = self._camera()
            self.camera_id = database.write_camera(camera)
            if self.imu_gravity_provider is not None:
                self.camera_sensor = pycolmap.sensor_t(
                    pycolmap.SensorType.CAMERA,
                    self.camera_id,
                )
                rig = pycolmap.Rig()
                rig.add_ref_sensor(self.camera_sensor)
                self.rig_id = database.write_rig(rig)
        finally:
            database.close()

    def add_image(self, image_name, image_id, timestamp_s):
        gravity, status = self._gravity(timestamp_s)
        database = pycolmap.Database.open(self.database_path)
        try:
            if self.camera_sensor is None:
                database.write_image(
                    pycolmap.Image(
                        image_id=image_id,
                        name=image_name,
                        camera_id=self.camera_id,
                    ),
                    use_image_id=True,
                )
                return image_id, status

            camera_data = pycolmap.data_t(self.camera_sensor, image_id)
            colmap_frame = pycolmap.Frame(rig_id=self.rig_id)
            colmap_frame.add_data_id(camera_data)
            colmap_frame.finalize_data_ids()
            frame_id = database.write_frame(colmap_frame)
            database.write_image(
                pycolmap.Image(
                    image_id=image_id,
                    name=image_name,
                    camera_id=self.camera_id,
                    frame_id=frame_id,
                ),
                use_image_id=True,
            )
            if gravity is not None:
                pose_prior = pycolmap.PosePrior()
                pose_prior.corr_data_id = camera_data
                pose_prior.gravity = gravity
                database.write_pose_prior(pose_prior)
            return image_id, status
        finally:
            database.close()

    def summary(self):
        if self.imu_gravity_provider is None:
            return None
        return self.imu_gravity_provider.summary()

    def _gravity(self, timestamp_s):
        if self.imu_gravity_provider is None:
            return None, ""
        gravity, diagnostics = (
            self.imu_gravity_provider.gravity_at_video_time(timestamp_s)
        )
        status = (
            f" | IMU gravity: {diagnostics['reason']}"
            f" (|a|={diagnostics.get('acceleration_magnitude_m_s2', np.nan):.3f} m/s^2, "
            f"|w|={np.degrees(diagnostics.get('gyroscope_magnitude_rad_s', np.nan)):.2f} deg/s)"
        )
        return gravity, status

    def _camera(self):
        distortion = self.distortion.reshape(-1)
        distortion = np.pad(distortion, (0, max(0, 8 - len(distortion))))
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        k1, k2, p1, p2, k3, k4, k5, k6 = distortion[:8]
        width, height = self.image_size
        camera = pycolmap.Camera(
            model="FULL_OPENCV",
            width=width,
            height=height,
            params=np.array(
                [fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6],
                dtype=float,
            ),
        )
        camera.has_prior_focal_length = True
        return camera
