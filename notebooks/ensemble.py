import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import json
import warnings
warnings.filterwarnings('ignore')

class FraudEnsemble:
    def __init__(self, artifacts_dir="./"):
        self.artifacts_dir = artifacts_dir
        self.models_config = {
            'lgbm': {
                'oof_path': 'lgbm_artifacts/lgbm_oof_predictions.npy',
                'test_path': 'lgbm_artifacts/lgbm_test_predictions.npy',
                'metadata_path': 'lgbm_artifacts/lgbm_metadata.json'
            },
            'xgb': {
                'oof_path': 'xgboost_artifacts/xgb_oof_predictions.npy',
                'test_path': 'xgboost_artifacts/xgb_test_predictions.npy',
                'metadata_path': 'xgboost_artifacts/xgb_metadata.json'
            },
            'catboost': {
                'oof_path': 'catboost_artifacts/catboost_oof_predictions.npy',
                'test_path': 'catboost_artifacts/catboost_test_predictions.npy',
                'metadata_path': 'catboost_artifacts/catboost_metadata.json'
            }
        }
        self.ensemble_weights = None
        self.meta_model = None
        
    def coerce_transaction_id(self, s: pd.Series) -> pd.Series:
        """Convert TransactionID to proper integer format"""
        if pd.api.types.is_integer_dtype(s): 
            return s.astype("int64")
        if pd.api.types.is_float_dtype(s):   
            return s.round().astype("int64")
        if pd.api.types.is_string_dtype(s):
            s2 = s.str.replace(r"\.0$", "", regex=True)
            return pd.to_numeric(s2, errors="raise").astype("int64")
        return pd.to_numeric(s, errors="raise").astype("int64")
    
    def load_predictions(self):
        """Load OOF and test predictions from all models"""
        print("Loading predictions from saved artifacts...")
        
        self.oof_predictions = {}
        self.test_predictions = {}
        self.model_scores = {}
        
        for model_name, paths in self.models_config.items():
            try:
                # Load OOF predictions
                oof_path = os.path.join(self.artifacts_dir, paths['oof_path'])
                if os.path.exists(oof_path):
                    self.oof_predictions[model_name] = np.load(oof_path)
                    print(f"✓ Loaded {model_name} OOF predictions: {self.oof_predictions[model_name].shape}")
                
                # Load test predictions
                test_path = os.path.join(self.artifacts_dir, paths['test_path'])
                if os.path.exists(test_path):
                    self.test_predictions[model_name] = np.load(test_path)
                    print(f"✓ Loaded {model_name} test predictions: {self.test_predictions[model_name].shape}")
                        
            except Exception as e:
                print(f"⚠️  Error loading {model_name}: {e}")
        
        print(f"\nSuccessfully loaded {len(self.oof_predictions)} models")
        return len(self.oof_predictions) > 0
    
    def calculate_cv_scores(self):
        """Calculate CV scores from OOF predictions after targets are loaded"""
        print("Calculating CV scores from OOF predictions...")
        
        for model_name in self.oof_predictions.keys():
            try:
                # Try to load from metadata first
                metadata_path = os.path.join(self.artifacts_dir, self.models_config[model_name]['metadata_path'])
                cv_score = 0
                
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        cv_score = metadata.get('cv_score', 0)
                
                # If CV score is 0 or missing, calculate from OOF predictions
                if cv_score == 0:
                    cv_score = roc_auc_score(self.y_true, self.oof_predictions[model_name])
                    print(f"✓ {model_name} CV Score (calculated): {cv_score:.4f}")
                else:
                    print(f"✓ {model_name} CV Score (from metadata): {cv_score:.4f}")
                
                self.model_scores[model_name] = cv_score
                
            except Exception as e:
                print(f"⚠️  Error calculating CV score for {model_name}: {e}")
                # Fallback: calculate directly
                cv_score = roc_auc_score(self.y_true, self.oof_predictions[model_name])
                self.model_scores[model_name] = cv_score
                print(f"✓ {model_name} CV Score (fallback): {cv_score:.4f}")
    
    def load_targets(self, data_dir="../data/processed"):
        """Load target values for ensemble training"""
        try:
            y_df = pd.read_csv(f"{data_dir}/IEEE_Target.csv")
            self.y_true = y_df['isFraud'].values
            print(f"✓ Loaded targets: {self.y_true.shape}, fraud rate: {self.y_true.mean():.4f}")
            return True
        except Exception as e:
            print(f"⚠️  Error loading targets: {e}")
            return False
    
    def simple_weighted_average(self, use_cv_weights=True):
        """Simple weighted average ensemble based on CV scores"""
        print("\n" + "="*60)
        print("SIMPLE WEIGHTED AVERAGE ENSEMBLE")
        print("="*60)
        
        if use_cv_weights and len(self.model_scores) > 0 and sum(self.model_scores.values()) > 0:
            # Weight by CV performance
            total_score = sum(self.model_scores.values())
            weights = {model: score/total_score for model, score in self.model_scores.items()}
            print("Using CV score-based weights")
        else:
            # Equal weights (fallback)
            weights = {model: 1/len(self.oof_predictions) for model in self.oof_predictions.keys()}
            print("Using equal weights (CV scores not available or all zero)")
        
        print("Ensemble weights:")
        for model, weight in weights.items():
            print(f"  {model}: {weight:.4f}")
        
        # Combine OOF predictions
        oof_ensemble = np.zeros_like(list(self.oof_predictions.values())[0])
        for model, preds in self.oof_predictions.items():
            oof_ensemble += weights[model] * preds
        
        # Combine test predictions
        test_ensemble = np.zeros_like(list(self.test_predictions.values())[0])
        for model, preds in self.test_predictions.items():
            test_ensemble += weights[model] * preds
        
        # Calculate ensemble CV score
        ensemble_auc = roc_auc_score(self.y_true, oof_ensemble)
        print(f"\nEnsemble OOF AUC: {ensemble_auc:.4f}")
        
        # Compare with individual models
        print("\nComparison with individual models:")
        for model, preds in self.oof_predictions.items():
            model_auc = roc_auc_score(self.y_true, preds)
            improvement = ensemble_auc - model_auc
            print(f"  {model}: {model_auc:.4f} (diff: {improvement:+.4f})")
        
        self.ensemble_weights = weights
        return oof_ensemble, test_ensemble, ensemble_auc
    
    def optimize_weights(self, method='scipy'):
        """Optimize ensemble weights using different methods"""
        print("\n" + "="*60)
        print("OPTIMIZING ENSEMBLE WEIGHTS")
        print("="*60)
        
        from scipy.optimize import minimize
        
        # Stack OOF predictions
        X_stack = np.column_stack(list(self.oof_predictions.values()))
        model_names = list(self.oof_predictions.keys())
        
        def objective(weights):
            weights = weights / np.sum(weights)  # Normalize
            ensemble_pred = np.dot(X_stack, weights)
            return -roc_auc_score(self.y_true, ensemble_pred)
        
        # Initial weights (equal)
        n_models = len(model_names)
        initial_weights = np.ones(n_models) / n_models
        
        # Constraints: weights sum to 1 and are non-negative
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]
        
        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        optimal_weights = result.x / np.sum(result.x)  # Ensure normalization
        
        # Create weighted ensemble
        oof_ensemble = np.dot(X_stack, optimal_weights)
        
        # Test predictions
        X_test_stack = np.column_stack(list(self.test_predictions.values()))
        test_ensemble = np.dot(X_test_stack, optimal_weights)
        
        ensemble_auc = roc_auc_score(self.y_true, oof_ensemble)
        
        print("Optimized weights:")
        weights_dict = {}
        for i, model in enumerate(model_names):
            weights_dict[model] = optimal_weights[i]
            print(f"  {model}: {optimal_weights[i]:.4f}")
        
        print(f"\nOptimized Ensemble OOF AUC: {ensemble_auc:.4f}")
        
        self.ensemble_weights = weights_dict
        return oof_ensemble, test_ensemble, ensemble_auc
    
    def meta_model_ensemble(self, meta_model_type='logistic'):
        """Train a meta-model on OOF predictions"""
        print("\n" + "="*60)
        print(f"META-MODEL ENSEMBLE ({meta_model_type.upper()})")
        print("="*60)
        
        # Stack OOF predictions as features
        X_meta = np.column_stack(list(self.oof_predictions.values()))
        model_names = list(self.oof_predictions.keys())
        
        print(f"Meta-model training data shape: {X_meta.shape}")
        print(f"Features: {model_names}")
        
        # Train meta-model
        if meta_model_type == 'logistic':
            self.meta_model = LogisticRegression(random_state=42, max_iter=1000)
        elif meta_model_type == 'rf':
            self.meta_model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError("meta_model_type must be 'logistic' or 'rf'")
        
        self.meta_model.fit(X_meta, self.y_true)
        
        # Get meta-model predictions
        oof_ensemble = self.meta_model.predict_proba(X_meta)[:, 1]
        
        # Test predictions
        X_test_meta = np.column_stack(list(self.test_predictions.values()))
        test_ensemble = self.meta_model.predict_proba(X_test_meta)[:, 1]
        
        ensemble_auc = roc_auc_score(self.y_true, oof_ensemble)
        print(f"\nMeta-model Ensemble OOF AUC: {ensemble_auc:.4f}")
        
        # Show feature importance if available
        if hasattr(self.meta_model, 'coef_'):
            print("\nMeta-model coefficients:")
            for i, model in enumerate(model_names):
                print(f"  {model}: {self.meta_model.coef_[0][i]:.4f}")
        elif hasattr(self.meta_model, 'feature_importances_'):
            print("\nMeta-model feature importances:")
            for i, model in enumerate(model_names):
                print(f"  {model}: {self.meta_model.feature_importances_[i]:.4f}")
        
        return oof_ensemble, test_ensemble, ensemble_auc
    
    def rank_average_ensemble(self):
        """Rank-based ensemble (average of ranks)"""
        print("\n" + "="*60)
        print("RANK AVERAGE ENSEMBLE")
        print("="*60)
        
        # Convert predictions to ranks
        oof_ranks = {}
        test_ranks = {}
        
        for model, preds in self.oof_predictions.items():
            oof_ranks[model] = pd.Series(preds).rank(pct=True).values
            
        for model, preds in self.test_predictions.items():
            test_ranks[model] = pd.Series(preds).rank(pct=True).values
        
        # Average ranks
        oof_ensemble = np.mean(list(oof_ranks.values()), axis=0)
        test_ensemble = np.mean(list(test_ranks.values()), axis=0)
        
        ensemble_auc = roc_auc_score(self.y_true, oof_ensemble)
        print(f"Rank Average Ensemble OOF AUC: {ensemble_auc:.4f}")
        
        return oof_ensemble, test_ensemble, ensemble_auc
    
    def compare_all_methods(self):
        """Compare all ensemble methods"""
        print("\n" + "="*80)
        print("ENSEMBLE COMPARISON")
        print("="*80)
        
        results = {}
        
        # 1. Simple average
        oof_simple, test_simple, auc_simple = self.simple_weighted_average(use_cv_weights=False)
        results['Simple Average'] = {
            'oof': oof_simple, 'test': test_simple, 'auc': auc_simple
        }
        
        # 2. CV-weighted average
        oof_weighted, test_weighted, auc_weighted = self.simple_weighted_average(use_cv_weights=True)
        results['CV Weighted'] = {
            'oof': oof_weighted, 'test': test_weighted, 'auc': auc_weighted
        }
        
        # 3. Optimized weights
        oof_opt, test_opt, auc_opt = self.optimize_weights()
        results['Optimized Weights'] = {
            'oof': oof_opt, 'test': test_opt, 'auc': auc_opt
        }
        
        # 4. Meta-model (Logistic)
        oof_meta_lr, test_meta_lr, auc_meta_lr = self.meta_model_ensemble('logistic')
        results['Meta-Model (LR)'] = {
            'oof': oof_meta_lr, 'test': test_meta_lr, 'auc': auc_meta_lr
        }
        
        # 5. Rank average
        oof_rank, test_rank, auc_rank = self.rank_average_ensemble()
        results['Rank Average'] = {
            'oof': oof_rank, 'test': test_rank, 'auc': auc_rank
        }
        
        # Summary table
        print("\n" + "="*60)
        print("ENSEMBLE RESULTS SUMMARY")
        print("="*60)
        print(f"{'Method':<20} {'OOF AUC':<12} {'Improvement':<12}")
        print("-" * 44)
        
        # Individual model scores for comparison
        print("Individual Models:")
        for model, score in self.model_scores.items():
            print(f"  {model:<18} {score:<12.4f} {'baseline':<12}")
        
        print("\nEnsemble Methods:")
        best_auc = max([r['auc'] for r in results.values()])
        best_method = None
        
        # Fix: Check if model_scores has values before calling max()
        if self.model_scores:
            best_individual = max(self.model_scores.values())
        else:
            best_individual = 0
        
        for method, result in results.items():
            auc = result['auc']
            improvement = auc - best_individual
            if auc == best_auc:
                best_method = method
            print(f"  {method:<18} {auc:<12.4f} {improvement:+.4f}")
        
        print(f"\n🏆 Best method: {best_method} (AUC: {best_auc:.4f})")
        
        return results, best_method
    
    def create_submission(self, method='best', submission_name=None):
        """Create submission file for the best ensemble method"""
        
        # Load test transaction IDs
        DATA_DIR = "../data/processed"
        try:
            test_df = pd.read_csv(f"{DATA_DIR}/IEEE_Test.csv")
            # Apply the coercion function to fix TransactionID format
            test_ids = self.coerce_transaction_id(test_df['TransactionID'])
        except:
            print("⚠️  Could not load test TransactionIDs, using indices")
            test_ids = np.arange(len(list(self.test_predictions.values())[0]))
        
        if method == 'best':
            # Run comparison to find best method
            results, best_method = self.compare_all_methods()
            test_preds = results[best_method]['test']
            ensemble_auc = results[best_method]['auc']
        else:
            # Use specific method
            if method == 'simple':
                _, test_preds, ensemble_auc = self.simple_weighted_average(use_cv_weights=False)
            elif method == 'weighted':
                _, test_preds, ensemble_auc = self.simple_weighted_average(use_cv_weights=True)
            elif method == 'optimized':
                _, test_preds, ensemble_auc = self.optimize_weights()
            elif method == 'meta':
                _, test_preds, ensemble_auc = self.meta_model_ensemble('logistic')
            elif method == 'rank':
                _, test_preds, ensemble_auc = self.rank_average_ensemble()
            best_method = method
        
        # Create submission
        submission = pd.DataFrame({
            'TransactionID': test_ids,
            'isFraud': test_preds
        })
        
        if submission_name is None:
            submission_name = f"submission_ensemble_{best_method.lower().replace(' ', '_')}_cv{ensemble_auc:.4f}.csv"
        
        submission_path = os.path.join(self.artifacts_dir, "submissions", submission_name)
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)
        submission.to_csv(submission_path, index=False)
        
        print(f"\n✅ Submission saved: {submission_path}")
        print(f"   Method: {best_method}")
        print(f"   OOF AUC: {ensemble_auc:.4f}")
        print(f"   Predictions range: [{test_preds.min():.6f}, {test_preds.max():.6f}]")
        
        return submission_path, ensemble_auc
    
    def analyze_model_correlations(self):
        """Analyze correlations between model predictions"""
        print("\n" + "="*60)
        print("MODEL CORRELATION ANALYSIS")
        print("="*60)
        
        # Create correlation matrix
        oof_df = pd.DataFrame(self.oof_predictions)
        correlation_matrix = oof_df.corr()
        
        print("OOF Prediction Correlations:")
        print(correlation_matrix.round(4))
        
        # Calculate diversity metrics
        print(f"\nAverage correlation: {correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean():.4f}")
        
        return correlation_matrix
    
    def save_ensemble_artifacts(self, best_method, ensemble_auc, submission_path):
        """Save ensemble metadata and artifacts"""
        ensemble_metadata = {
            'best_method': best_method,
            'ensemble_auc': ensemble_auc,
            'individual_scores': self.model_scores,
            'weights': self.ensemble_weights if self.ensemble_weights else {},
            'submission_path': submission_path,
            'models_used': list(self.oof_predictions.keys())
        }
        
        metadata_path = os.path.join(self.artifacts_dir, "ensemble_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(ensemble_metadata, f, indent=2)
        
        print(f"✅ Ensemble metadata saved: {metadata_path}")

def main():
    """Main execution function"""
    print("🚀 Starting IEEE CIS Fraud Detection Ensemble")
    print("="*80)
    
    # Initialize ensemble
    ensemble = FraudEnsemble()
    
    # Load all predictions and targets
    if not ensemble.load_predictions():
        print("❌ Failed to load predictions. Check your artifacts directory.")
        return
    
    if not ensemble.load_targets():
        print("❌ Failed to load targets. Check your data directory.")
        return
    
    # CRITICAL FIX: Calculate CV scores after loading targets
    ensemble.calculate_cv_scores()
    
    # Analyze model correlations
    ensemble.analyze_model_correlations()
    
    # Create submissions for ALL ensemble methods
    print("\n" + "="*80)
    print("CREATING ALL ENSEMBLE SUBMISSIONS")
    print("="*80)
    
    ensemble_methods = ['simple', 'weighted', 'optimized', 'meta', 'rank']
    submission_paths = {}
    
    for method in ensemble_methods:
        try:
            submission_path, ensemble_auc = ensemble.create_submission(method=method)
            submission_paths[method] = submission_path
            print(f"✅ {method.upper()} submission: {os.path.basename(submission_path)}")
        except Exception as e:
            print(f"❌ Error creating {method} submission: {e}")
    
    # Compare all methods and identify best
    results, best_method = ensemble.compare_all_methods()
    
    # Save ensemble artifacts
    best_auc = max([r['auc'] for r in results.values()])
    best_submission = submission_paths.get(best_method.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_'))
    ensemble.save_ensemble_artifacts(best_method, best_auc, best_submission)
    
    print("\n🎉 Ensemble complete!")
    print(f"All submissions created in ./submissions/")
    print(f"🏆 Best submission: {os.path.basename(best_submission) if best_submission else 'N/A'}")

if __name__ == "__main__":
    main()