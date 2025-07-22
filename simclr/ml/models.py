# simclr/ml/models.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import matplotlib.cm as cm
from sklearn.decomposition import PCA
import cv2
import logging
import traceback

# 로거 설정
logger = logging.getLogger(__name__)

# SimCLR 모델 정의
class SimCLR(nn.Module):
    def __init__(self, base_model='resnet18', projection_dim=128):
        super(SimCLR, self).__init__()
        resnet = getattr(models, base_model)(weights=None)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x).squeeze()
        # 1D 텐서인 경우 처리
        if h.dim() == 1:
            h = h.unsqueeze(0)
        z = self.projection(h)
        return F.normalize(z, dim=-1)

# Mahalanobis 거리 계산기
class MahalanobisDistanceCalculator:
    def __init__(self, reference_features):
        """
        Mahalanobis 거리 계산기 초기화

        Args:
            reference_features: 정상 샘플의 특징 벡터들 (N x D)
        """
        self.ref_feats = reference_features
        self.mean = np.mean(reference_features, axis=0)

        # 차원이 높은 경우 PCA로 차원 축소
        if reference_features.shape[1] > 100:
            try:
                n_components = min(100, reference_features.shape[0] - 1)
                pca = PCA(n_components=n_components)
                self.ref_feats_reduced = pca.fit_transform(reference_features)
                self.mean_reduced = np.mean(self.ref_feats_reduced, axis=0)
                cov = np.cov(self.ref_feats_reduced, rowvar=False)
                self.pca = pca
                self.use_pca = True
                logger.info(f"PCA 차원 축소 적용: {reference_features.shape[1]} -> {n_components}")
            except Exception as e:
                logger.warning(f"PCA 적용 실패, 원본 특징 사용: {str(e)}")
                cov = np.cov(reference_features, rowvar=False)
                self.use_pca = False
        else:
            cov = np.cov(reference_features, rowvar=False)
            self.use_pca = False

        # 공분산 행렬의 역행렬 계산 (특이값 분해를 사용하여 안정성 향상)
        try:
            U, S, Vh = np.linalg.svd(cov)
            
            # 수치적 안정성을 위해 매우 작은 특이값 처리
            epsilon = 1e-6
            S_inv = np.diag(1.0 / (S + epsilon))
            self.cov_inv = U @ S_inv @ Vh
            
            # 역행렬 유효성 검사
            if np.any(np.isnan(self.cov_inv)) or np.any(np.isinf(self.cov_inv)):
                raise ValueError("공분산 역행렬에 NaN 또는 Inf 값이 있습니다.")
                
            logger.info("공분산 역행렬 계산 성공")
        except Exception as e:
            logger.error(f"공분산 역행렬 계산 실패: {str(e)}")
            # 문제 시 단위행렬로 대체
            d = reference_features.shape[1] if not self.use_pca else self.ref_feats_reduced.shape[1]
            self.cov_inv = np.eye(d)
            logger.warning("단위행렬로 대체하여 계속 진행")

    def calculate_distance(self, feature):
        """
        특징 벡터와 참조 분포 사이의 Mahalanobis 거리 계산

        Args:
            feature: 입력 특징 벡터 (D,)

        Returns:
            distance: Mahalanobis 거리
        """
        try:
            if self.use_pca:
                feature_reduced = self.pca.transform(feature.reshape(1, -1)).reshape(-1)
                diff = feature_reduced - self.mean_reduced
            else:
                diff = feature - self.mean

            distance = np.sqrt(max(diff @ self.cov_inv @ diff.T, 0))  # 음수 방지
            
            # 유효한 거리값 확인
            if np.isnan(distance) or np.isinf(distance):
                logger.warning("유효하지 않은 거리값 (NaN 또는 Inf) 발생, 유클리드 거리로 대체")
                # 유클리드 거리로 대체
                if self.use_pca:
                    distance = np.linalg.norm(feature_reduced - self.mean_reduced)
                else:
                    distance = np.linalg.norm(feature - self.mean)
                    
            return distance
        except Exception as e:
            logger.error(f"거리 계산 오류: {str(e)}")
            # 오류 발생 시 유클리드 거리 반환
            return np.linalg.norm(feature - self.mean)

# 앙상블 모델
class MultiScaleModelEnsemble:
    def __init__(self, model_paths, patch_sizes, base_model='resnet18', device='cuda', weights=None):
        """
        다양한 패치 크기의 모델들을 앙상블하는 클래스

        Args:
            model_paths: 각 모델의 가중치 경로 리스트
            patch_sizes: 각 모델의 패치 크기 리스트
            base_model: 기본 모델 아키텍처
            device: 연산 장치
            weights: 각 모델의 가중치 (None이면 동일 가중치 적용)
        """
        self.models = []
        self.patch_sizes = patch_sizes
        self.device = device
        self.weights = weights if weights else [1.0 / len(patch_sizes)] * len(patch_sizes)
        
        logger.info(f"MultiScaleModelEnsemble 초기화 - 패치 크기: {patch_sizes}, 장치: {device}")

        # 각 패치 크기별 모델 로드
        for model_path, patch_size in zip(model_paths, patch_sizes):
            logger.info(f"모델 로드 중: {model_path}")
            
            # 모델 파일 존재 확인
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"모델 파일이 존재하지 않습니다: {model_path}")
                
            # 모델 생성 및 가중치 로드
            model = SimCLR(base_model=base_model).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            self.models.append(model)
            logger.info(f"모델 로드 완료: {patch_size}px")
            
        logger.info("모든 모델 로드 완료")

    def generate_anomaly_map(self, image_path, reference_feature_dirs):
        """
        앙상블 이상 맵 생성

        Args:
            image_path: 분석할 이미지 경로
            reference_feature_dirs: 각 모델의 참조 특징 디렉토리 리스트

        Returns:
            ensemble_anomaly_map: 앙상블된 이상 맵
            img: 원본 이미지 (PIL)
        """
        logger.info(f"이미지 분석 시작: {image_path}")

        # 원본 이미지 로드
        try:
            img = Image.open(image_path).convert('RGB')
            img_np = np.array(img)
            H, W, _ = img_np.shape
            logger.info(f"이미지 로드 성공: {H}x{W}px")
        except Exception as e:
            logger.exception(f"이미지 로드 실패: {str(e)}")
            raise

        # 통합 이상 맵 초기화
        ensemble_anomaly_map = np.zeros((H, W))

        # 각 모델별 이상 맵 생성 및 통합
        for i, (model, patch_size, ref_dir) in enumerate(zip(self.models, self.patch_sizes, reference_feature_dirs)):
            logger.info(f"모델 {i+1}/{len(self.models)} (패치 크기: {patch_size}px) 처리 중")

            # 개별 이상 맵 생성
            try:
                anomaly_map = self._generate_single_anomaly_map(
                    image_path,
                    patch_size,
                    model,
                    ref_dir
                )
                
                # 필요한 경우 크기 조정
                if anomaly_map.shape != (H, W):
                    logger.info(f"이상 맵 크기 조정: {anomaly_map.shape} -> {(H, W)}")
                    anomaly_map = cv2.resize(anomaly_map, (W, H))

                # 0~1로 정규화
                if np.ptp(anomaly_map) > 1e-8:  # 분모가 0이 되는 것 방지
                    anomaly_map = (anomaly_map - anomaly_map.min()) / np.ptp(anomaly_map)
                else:
                    anomaly_map = np.zeros_like(anomaly_map)
                    logger.warning(f"모델 {i+1} 이상 맵 정규화 실패: 값 범위가 너무 작음")

                # 가중치 적용 및 통합
                ensemble_anomaly_map += self.weights[i] * anomaly_map
                logger.info(f"모델 {i+1} 이상 맵 통합 완료 (가중치: {self.weights[i]})")
            
            except Exception as e:
                logger.exception(f"모델 {i+1} 이상 맵 생성 실패: {str(e)}")
                # 한 모델이 실패해도 계속 진행
                continue

        # 최종 앙상블 맵 정규화
        if np.ptp(ensemble_anomaly_map) > 1e-8:
            ensemble_anomaly_map = (ensemble_anomaly_map - ensemble_anomaly_map.min()) / np.ptp(ensemble_anomaly_map)
        else:
            # 모든 모델이 실패한 경우
            logger.error("모든 모델이 유효한 이상 맵을 생성하지 못했습니다.")
            ensemble_anomaly_map = np.zeros((H, W))
        
        logger.info("앙상블 이상 맵 생성 완료")
        return ensemble_anomaly_map, img

    def _generate_single_anomaly_map(self, image_path, patch_size, model, reference_features_dir):
        """
        단일 모델을 사용한 이상 맵 생성

        Args:
            image_path: 분석할 이미지 경로
            patch_size: 패치 크기
            model: 사용할 모델
            reference_features_dir: 참조 특징 디렉토리

        Returns:
            anomaly_map: 생성된 이상 맵
        """
        logger.info(f"단일 모델({patch_size}px) 이상 맵 생성 시작")
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # 이미지 로드
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
        H, W, _ = img_np.shape

        # 이미지 크기 검증
        if H < patch_size or W < patch_size:
            logger.warning(f"이미지 크기({H}x{W})가 패치 크기({patch_size})보다 작습니다. 이미지 크기 조정 중...")
            scale_factor = max(patch_size / H, patch_size / W) * 1.1  # 여유있게 10% 더 크게
            new_h, new_w = int(H * scale_factor), int(W * scale_factor)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            img_np = np.array(img)
            H, W, _ = img_np.shape
            logger.info(f"이미지 크기 조정 완료: {H}x{W}px")

        # 패치 추출 설정
        stride = max(patch_size // 4, 1)  # 최소 스트라이드 보장
        anomaly_map = np.zeros((H, W))
        count_map = np.zeros((H, W))

        # 참조 특징 로드
        try:
            ref_feats = []
            feature_files = [f for f in os.listdir(reference_features_dir) if f.endswith('_features.npy')]
            
            if not feature_files:
                raise FileNotFoundError(f"참조 특징 파일이 없습니다: {reference_features_dir}")
            
            logger.info(f"참조 특징 파일 {len(feature_files)}개 발견")
            
            for f in feature_files:
                ref_path = os.path.join(reference_features_dir, f)
                ref = np.load(ref_path)
                if ref.ndim == 2:
                    ref_feats.append(ref)
            
            if not ref_feats:
                raise ValueError("유효한 참조 특징 데이터가 없습니다.")
                
            ref_feats = np.concatenate(ref_feats, axis=0)
            logger.info(f"참조 특징 로드 완료: {ref_feats.shape}")
        except Exception as e:
            logger.exception(f"참조 특징 로드 실패: {str(e)}")
            raise

        # Mahalanobis 거리 계산기 초기화
        try:
            mahalanobis_calc = MahalanobisDistanceCalculator(ref_feats)
            logger.info("Mahalanobis 거리 계산기 초기화 완료")
        except Exception as e:
            logger.exception(f"Mahalanobis 거리 계산기 초기화 실패: {str(e)}")
            raise

        # 패치 특징 추출 및 비교
        try:
            with torch.no_grad():
                total_patches = ((H - patch_size) // stride + 1) * ((W - patch_size) // stride + 1)
                processed = 0
                
                for y in range(0, H - patch_size + 1, stride):
                    for x in range(0, W - patch_size + 1, stride):
                        # 진행 상황 로깅 (10% 단위)
                        processed += 1
                        if processed % max(1, total_patches // 10) == 0:
                            logger.info(f"패치 처리 중: {processed}/{total_patches} ({processed/total_patches*100:.1f}%)")
                        
                        # 패치 추출 및 특징 추출
                        patch = img_np[y:y+patch_size, x:x+patch_size, :]
                        patch_tensor = transform(Image.fromarray(patch)).unsqueeze(0).to(self.device)
                        feat = model(patch_tensor).cpu().numpy().reshape(-1)

                        # Mahalanobis 거리 계산
                        anomaly_score = mahalanobis_calc.calculate_distance(feat)

                        # 이상 점수가 NaN이나 Inf인 경우 처리
                        if np.isnan(anomaly_score) or np.isinf(anomaly_score):
                            logger.warning(f"유효하지 않은 이상 점수: {anomaly_score}, 0으로 대체")
                            anomaly_score = 0.0

                        # 이상 점수 맵에 추가
                        anomaly_map[y:y+patch_size, x:x+patch_size] += anomaly_score
                        count_map[y:y+patch_size, x:x+patch_size] += 1
        
            # 평균 계산 (분모가 0인 경우 0으로 처리)
            anomaly_map = np.divide(anomaly_map, count_map, out=np.zeros_like(anomaly_map), where=count_map!=0)
            anomaly_range = np.ptp(anomaly_map)
            if anomaly_range < 1e-8:
                logger.warning(f"모델 이상 맵 정규화 실패: 값 범위가 너무 작음 ({anomaly_range})")
         # 작은 임의의 노이즈 추가하여 범위 확장
                np.random.seed(42)  # 결과 재현성 위해
                noise = np.random.normal(0, 0.01, anomaly_map.shape)
                anomaly_map = anomaly_map + noise
                logger.info(f"노이즈 추가 후 범위: {np.ptp(anomaly_map):.6f}")
            return anomaly_map
        
        except Exception as e:
            logger.exception(f"패치 처리 중 오류 발생: {str(e)}")
            raise