import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TrajectoryGoalDataset(Dataset):
    """
    Датасет, который принимает путь к .npy файлу и сам делает всю обработку:
      - загружает траектории (N, L, 3)
      - берёт первые `points_to_take` точек
      - прореживает с шагом `thinning_factor`
      - нормализует в [-1, 1] глобально по всему датасету (0 остаётся 0)
      - возвращает {'trajectory': (seq_len, 3), 'goal': (3,)}
    """
    def __init__(
        self,
        npy_path: str,
        points_to_take: int = 40,
        thinning_factor: int = 2,
        normalize: bool = True,
        verbose: bool = True
    ):
        if verbose:
            print(f"Загружаем: {npy_path}")
        
        # 1. Чтение файла
        try:
            traj_full = np.load(npy_path)
        except Exception as e:
            raise ValueError(f"Не удалось загрузить файл {npy_path}\nОшибка: {e}")
        
        if traj_full.ndim != 3 or traj_full.shape[2] != 3:
            raise ValueError(f"Ожидается массив (N, L, 3), получено shape {traj_full.shape}")
        
        N, L, _ = traj_full.shape
        
        if L < points_to_take:
            raise ValueError(f"Траектории слишком короткие: {L} < {points_to_take}")
        
        if verbose:
            print(f"Найдено {N} траекторий, длина {L}")
        
        # 2. Обрезка + прореживание
        traj_short = traj_full[:, :points_to_take, :]
        self.trajectories = traj_short[:, ::thinning_factor, :]   # (N, seq_len, 3)
        self.goals = traj_full[:, -1, :]                          # (N, 3)
        
        self.seq_len = self.trajectories.shape[1]
        
        if verbose:
            print(f"После обработки → траектории: {self.trajectories.shape}, цели: {self.goals.shape}")
        
        # 3. Нормализация в [-1, 1] (0 остаётся 0)
        if normalize:
            all_points = np.concatenate([
                self.trajectories.reshape(-1, 3),
                self.goals
            ], axis=0)
            
            self.scales = np.max(np.abs(all_points), axis=0)  # (3,)
            
            if verbose:
                print("Масштабы (max_abs по осям):", np.round(self.scales, 4))
            
            self.trajectories = self.trajectories / self.scales[None, None, :]
            self.goals = self.goals / self.scales[None, :]
            
            if verbose:
                print("После нормализации min/max траекторий по осям:",
                      np.round(self.trajectories.min(axis=(0,1)), 3),
                      np.round(self.trajectories.max(axis=(0,1)), 3))
        else:
            self.scales = np.ones(3, dtype=np.float32)
        
        # 4. В torch тензоры
        self.trajectories = torch.from_numpy(self.trajectories).float()
        self.goals = torch.from_numpy(self.goals).float()

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return {
            'trajectory': self.trajectories[idx],   # (seq_len, 3)
            'goal': self.goals[idx],                # (3,)
        }

    def get_scales(self):
        """Возвращает коэффициенты масштабирования для денормализации"""
        return self.scales
    


if __name__ == "__main__":
    # Путь к файлу
    path = "track_dataset_zero_init.npy"          # ← подставь свой путь

    dataset = TrajectoryGoalDataset(
        npy_path=path,
        points_to_take=40,
        thinning_factor=2,
        normalize=True,
        verbose=True
    )

    print(dataset[0])