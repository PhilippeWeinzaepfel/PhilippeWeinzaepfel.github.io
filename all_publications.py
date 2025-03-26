import sys, os, pdb
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

publication_list = [
    2025,
    {
        'teaser': 'dune.jpg',
        'title': 'DUNE: Distilling a Universal Encoder from Heterogeneous 2D and 3D Teachers',
        'authors': 'Mert Bulent Sariyildiz, Philippe Weinzaepfel, Thomas Lucas, Pau de Jorge, Diane Larlus, Yannis Kalantidis',
        'where': 'CVPR 2025',
        'arxiv': 'https://arxiv.org/abs/2503.14405',
        'project': 'https://europe.naverlabs.com/research/publications/dune/',
        'tldr': 'A single encoder distilled from multiple teachers: DINOv2, MASt3R and Multi-HMR, versatile enough to perform heterogeneous tasks.',
        'star': True,
    },
    {
        'teaser': 'pow3r.jpg',
        'title': 'Pow3R: Empowering Unconstrained 3D Reconstruction with Camera and Scene Priors',
        'authors': 'Wonbong Jang, Philippe Weinzaepfel, Vincent Leroy, Lourdes Agapito, Jerome Revaud',
        'where': 'CVPR 2025',
        'arxiv': 'https://arxiv.org/abs/2503.17316',
        'project': 'https://europe.naverlabs.com/pow3r',
        'tldr': 'Versatile integration of several camera and scene priors into DUSt3R-like approaches.',
        'star': True,
    },
    {
        'teaser': 'mast3rsfm.jpg',
        'title': 'MASt3R-SfM: a Fully-Integrated Solution for Unconstrained Structure-from-Motion',
        'authors': 'Bardienus Duisterhof, Lojze Zust, Philippe Weinzaepfel, Vincent Leroy, Yohann Cabon, Jerome Revaud',
        'where': '3DV 2025 (oral) Best Student Paper Award <span style="padding-left: 5px;"><svg class="button-link-icon"><use href="#winner-icon"></use></svg></span>',
        'arxiv': 'https://arxiv.org/abs/2409.19152',
        'github': 'https://github.com/naver/mast3r',
        'tldr': 'Scaling MASt3R to large image collections thanks to using the encoder features for retrieval and a sparse alignment formulation.',
        'star': True,
    },
    2024,
    {
        'teaser': 'condimen.jpg',
        'title': 'CondiMen: Conditional Multi-Person Mesh Recovery',
        'authors': 'Romain Brégier, Fabien Baradel, Thomas Lucas, Salma Galaaoui, Matthieu Armando, Philippe Weinzaepfel, Grégory Rogez',
        'where': 'arXiv 2024',
        'arxiv': 'https://arxiv.org/abs/2412.13058',
        'tldr': 'A multi-person human mesh recovery method that outputs a joint parametric distribution over likely poses, body shapes, intrinsics and distances to the camera, using a Bayesian network.',
    },
	{
		'teaser': 'MultiHMR.gif',
		'title': 'Multi-HMR: Multi-Person Whole-Body Human Mesh Recovery in a Single Shot',
		'authors': 'Fabien Baradel, Matthieu Armando, Salma Galaaoui, Romain Brégier, Philippe Weinzaepfel, Grégory Rogez, Thomas Lucas',
		'where': 'ECCV 2024',
		'arxiv': 'https://arxiv.org/abs/2402.14654',
		'project': 'https://europe.naverlabs.com/blog/whole-body-human-mesh-recovery-of-multiple-persons-from-a-single-image/',
		'github': 'https://github.com/naver/multi-hmr',
		'huggingface': 'https://huggingface.co/spaces/naver/multi-hmr',
		'winner': ('https://rhobin-challenge.github.io/', "winner of ROBIN challenge @CVPR'24"),
		'tldr': 'A simple yet effective model for multi-person whole-body human mesh recovery estimation running in real-time on a GPU and reaching SotA results.',
        'star': True,
        'confdemo': 'ECCV 2024',
	},
	{
		'teaser': 'PoseEmbroider.jpg',
		'title': 'PoseEmbroider: Towards a 3D, Visual, Semantic-Aware Human Pose Representation',
		'authors': 'Ginger Delmas, Philippe Weinzaepfel, Francesc Moreno-Noguer, Grégory Rogez',
		'where': 'ECCV 2024',
		'arxiv': 'https://arxiv.org/abs/2409.06535',
		'github': 'https://github.com/naver/poseembroider',
		'project': 'https://europe.naverlabs.com/research/publications/poseembroider-towards-a-3d-visual-semantic-aware-human-pose-representation/',
		'tldr': "A multi-modal representation with any combination of modalities among human pose, text description and person's picture.",
        'star': True,
	},
	{
		'teaser': 'unic.png',
		'title': 'UNIC: Universal Classification Models via Multi-teacher Distillation',
		'authors': 'Mert Bulent Sariyildiz, Philippe Weinzaepfel, Thomas Lucas, Diane Larlus, Yannis Kalantidis',
		'where': 'ECCV 2024',
        'arxiv': 'https://arxiv.org/abs/2408.05088',
        'github': 'https://github.com/naver/unic',
        'project': 'https://europe.naverlabs.com/research/publications/unic-universal-classification-models-via-multi-teacher-distillation/',
        'tldr': 'A UNIC classification model that distills from strong pretrained models, and performs on par of better then each of them.',
        'star': True,
	},
    {
        'teaser': 'CroCoMan.png',
        'title': 'Cross-view and Cross-pose Completion for 3D Human Understanding',
        'authors': 'Matthieu Armando, Salma Galaaoui, Fabien Baradel, Thomas Lucas, Vincent Leroy, Romain Brégier, Philippe Weinzaepfel, Grégory Rogez',
        'where': 'CVPR 2024',
        'arxiv': 'https://arxiv.org/abs/2311.09104',
        'project': 'https://europe.naverlabs.com/research/publications/cross-view-and-cross-pose-completion-for-3d-human-understanding/',
        'tldr': 'Applying the CroCo pre-training idea on human-centric data, sampling image pairs from multi-view or video datasets.', 
    },
    {
        'teaser': 'SACReg.jpg',
        'title': 'SACReg: Scene-agnostic coordinate regression for visual localization',
        'authors': 'Jerome Revaud, Yohann Cabon, Romain Brégier, JongMin Lee, Philippe Weinzaepfel',
        'where': 'CVPR Workshop 2024',
        'arxiv': 'https://arxiv.org/abs/2307.11702',
        'tldr': 'Masking scene coordinate regression models scene-agnostic by considering the 2D-3D matches as an external database.'    
    },
    {
        'teaser': 'CroCoNav.jpg',
        'title': 'End-to-End (Instance)-Image Goal Navigation Through Correspendence As An Emerging Phenomenon',
        'authors': 'Guillaume Bono, Leonid Antsfeld, Boris Chidlovskii, Philippe Weinzaepfel, Christian Wolf',
        'where': 'ICLR 2024',
        'arxiv': 'https://arxiv.org/abs/2309.16634',
        'tldr': ' In an ImageGoal navigation context, we propose two pre-text tasks which let correspondence emerge as a solution and train a dual visual encoder based on a binocular transformer.',
        'star': True,
    },
    {
        'teaser': 'WinWin.jpg',
        'title': 'Win-Win: Training High-Resolution Vision Transformers from Two Windows',
        'authors': 'Vincent Leroy, Jerome Revaud, Thomas Lucas, Philippe Weinzaepfel',
        'where': 'ICLR 2024',
        'arxiv': 'https://arxiv.org/abs/2310.00632',
        'tldr': 'Win-Win enables to efficiently train vanilla ViTs for high-resolution dense pixelwise tasks.',
        'star': True,
    },
    {
        'teaser': 'Ret4Loc.jpg',
        'title': 'Weatherproofing Retrieval for Localization with Generative AI and Geometric Consistency',
        'authors': 'Yannis Kalantidis, Mert Bulent Sariyildiz, Rafael S Rezende, Philippe Weinzaepfel, Diane Larlus, Gabriela Csurka',
        'where': 'ICLR 2024',
        'arxiv': 'https://arxiv.org/abs/2402.09237',
        'project': 'https://europe.naverlabs.com/research/publications/weatherproofing-retrieval-for-localization-with-generative-ai-and-geometric-consistency/',
        'tldr': 'We make retrieval for localization models robust to weather, seasonal and time-of-day changes by augmenting the training set with synthetic variations generated using Generative AI and leverage geometric consistency for sampling and filtering.',
    },
    {
        'teaser': 'SHOWMe2.jpg',
        'title': 'SHOWMe: Robust object-agnostic hand-object 3D reconstruction from RGB video',
        'authors': 'Anilkumar Swamy, Vincent Leroy, Philippe Weinzaepfel, Fabien Baradel, Salma Galaaoui, Romain Brégier, Matthieu Armando, Jean-Sebastien Franco, Grégory Rogez',
        'where': 'CVIU 2024',
        'paper': 'https://www.sciencedirect.com/science/article/abs/pii/S1077314224001541?via%3Dihub',
        'dataset': 'https://download.europe.naverlabs.com/showme/',
        'tldr': 'Extension of SHOWMe (ICCVW 2023) with improved hand-object reconstruction by extending the two-stage method with the estimation of virtual camera poses based on a finetuned CroCo model.',
        'star': True,
    },
    {
        'teaser': 'purposer.jpg',
        'title': 'Purposer: Putting Human Motion Generation in Context',
        'authors': 'Nicolás Ugrinovic, Thomas Lucas, Fabien Baradel, Philippe Weinzaepfel, Gregory Rogez, Francesc Moreno-Noguer',
        'where': '3DV 2024',
        'arxiv': 'https://arxiv.org/abs/2404.12942',
        'tldr': 'A versatile method able to generate realistic-looking motions that interact with virtual scenes.',
    },
    {
        'teaser': 'posescript2.jpg',
        'title':  'PoseScript: Linking 3D Human Poses and Natural Language',
        'authors': 'Ginger Delmas, Philippe Weinzaepfel, Thomas Lucas, Francesc Moreno-Noguer, Grégory Rogez',
        'where': 'IEEE Trans. PAMI 2024',
        'arxiv': 'https://arxiv.org/abs/2210.11795',
        'github': 'https://github.com/naver/posescript',
        'project': 'https://europe.naverlabs.com/research/publications/posescript-3d-human-poses-from-natural-language/',
        'tldr': 'Extension of the ECCV 2022 paper on dataset and tasks that relate 3D human poses and their description in natural language.',
    },
    2023,
    {
        'teaser': 'CroCo_v2.jpg',
        'title': 'CroCo v2: Improved Cross-view Completion Pre-training for Stereo Matching and Optical Flow',
        'authors': 'Philippe Weinzaepfel, Thomas Lucas, Vincent Leroy, Yohann Cabon, Vaibhav Arora, Romain Brégier, Gabriela Csurka, Leonid Antsfeld, Boris Chidlovskii, Jérôme Revaud',
        'where': 'ICCV 2023',
        'arxiv': 'https://arxiv.org/abs/2211.10408',
        'github': 'https://github.com/naver/croco',
        'tldr': 'Extending CroCo to real datasets leads to state-of-the-art results for binocular tasks like stereo and flow.',
        'star': True,
    },
    {
        'teaser': 'posefix.jpg',
        'title': 'PoseFix: Correcting 3D Human Poses with Natural Language',
        'authors': 'Ginger Delmas, Philippe Weinzaepfel, Francesc Moreno-Noguer, Grégory Rogez',
        'where': 'ICCV 2023',
        'arxiv': 'https://arxiv.org/abs/2309.08480',
        'github': 'https://github.com/naver/posescript',
        'dataset': 'https://download.europe.naverlabs.com/ComputerVision/PoseFix/posefix_dataset_release.zip',
        'tldr': 'Text description of the difference between two human poses.',
        'star': True,
    },
    {
        'teaser': 'SHOWMe.jpg',
        'title': 'SHOWMe: Benchmarking object-agnostic hand-object 3d reconstruction',
        'authors': 'Anilkumar Swamy, Vincent Leroy, Philippe Weinzaepfel, Fabien Baradel, Salma Galaaoui, Romain Brégier, Matthieu Armando, Jean-Sebastien Franco, Grégory Rogez',
        'where': 'ICCV Workshop 2023',
        'arxiv': 'https://arxiv.org/abs/2309.10748',
        'github': 'https://github.com/naver/showme',
        'project': 'https://europe.naverlabs.com/research/showme/',
        'tldr': 'A dataset and method for highly-detailed object-agnostic hand-object reconstruction',
    },
    2022,
    {
        'teaser': 'CroCo.jpg',
        'title': 'CroCo: Self-Supervised Pre-Training for 3D Vision Tasks by Cross-view Completion',
        'authors': 'Philippe Weinzaepfel, Vincent Leroy, Thomas Lucas, Romain Brégier, Yohann Cabon, Vaibhav Arora, Leonid Antsfeld, Boris Chidlovskii, Gabriela Csurka, Jérôme Revaud',
        'where': 'NeurIPS 2022',
        'arxiv': 'https://arxiv.org/abs/2210.10716',
        'github': 'https://github.com/naver/croco',
        'project': 'https://europe.naverlabs.com/research/publications/croco-self-supervised-pretraining-for-3d-vision-tasks-by-cross-view-completion/',
        'tldr': 'Masked image modeling with a second reference view implicitly learns correspondences and is thus well suited for geometric tasks.',
        'star': True,
    },
    {
        'teaser': 'posebert.jpg',
        'title': 'PoseBERT: A Generic Transformer Module for Temporal 3D Human Modeling',
        'authors': 'Fabien Baradel, Romain Brégier, Thibault Groueix, Philippe Weinzaepfel, Yannis Kalantidis, Grégory Rogez',
        'where': 'IEEE Trans. PAMI 2022',
        'arxiv': 'https://arxiv.org/abs/2208.10211',
        'github': 'https://github.com/naver/posebert',
        'tldr': 'A generic transformer model for temporal modeling of human and hand shape trained with masked modeling and that can be applied e.g. to pose estimation and future pose prediction (extension of our 3DV 2021 paper).',
    },
    {
        'teaser': 'posescript.jpg',
        'title':  'PoseScript: Linking 3D Human Poses and Natural Language',
        'authors': 'Ginger Delmas, Philippe Weinzaepfel, Thomas Lucas, Francesc Moreno-Noguer, Grégory Rogez',
        'where': 'ECCV 2022',
        'arxiv': 'https://arxiv.org/abs/2210.11795',
        'github': 'https://github.com/naver/posescript',
        'project': 'https://europe.naverlabs.com/research/publications/posescript-3d-human-poses-from-natural-language/',
        'tldr': 'Dataset and tasks that relate 3D human poses and their description in natural language.',
        'star': True,
    },
    {
        'teaser': 'posegpt.png',
        'title': 'PoseGPT: Quantization-based 3D Human Motion Generation and Forecasting',
        'authors': 'Thomas Lucas, Fabien Baradel, Philippe Weinzaepfel, Grégory Rogez',
        'where': 'ECCV 2022',
        'arxiv': 'https://arxiv.org/abs/2210.10542',
        'github': 'https://github.com/naver/PoseGPT',
        'tldr': 'PoseGPT generates a human motion, conditioned on an action label, a duration and optionally on an observed past human motion using a VQ-VAE.',
    },
    {
        'teaser': 'grasplikehumans.jpg',
        'title': 'Multi-Finger Grasping Like Humans',
        'authors': 'Yuming Du, Philippe Weinzaepfel, Vincent Lepetit, Romain Brégier',
        'where': 'IROS 2022',
        'arxiv': 'https://arxiv.org/abs/2211.07304',
        'tldr': 'An optimization-based approach to transform a human grasp into a multi-finger robot grasp.',
    },
    {
        'teaser': 'retrievalkapture.jpg',
        'title': 'Investigating the role of image retrieval for visual localization: An exhaustive benchmark',
        'authors': 'Martin Humenberger, Yohann Cabon, Noé Pion, Philippe Weinzaepfel, Donghwan Lee, Nicolas Guérin, Torsten Sattler, Gabriela Csurka',
        'where': 'IJCV 2022',
        'arxiv': 'https://arxiv.org/abs/2205.15761',
        'github': 'https://github.com/naver/kapture-localization',
        'tldr': 'We analyze the role of image retrieval for three visual localization paradigms.',
    },
    {
        'teaser': 'pump.jpg',
        'title': 'PUMP: Pyramidal and Uniqueness Matching Priors for Unsupervised Learning of Local Descriptors',
        'authors': 'Jérome Revaud, Vincent Leroy, Philippe Weinzaepfel, Boris Chidlovskii',
        'where': 'CVPR 2022',
        'paper': 'https://openaccess.thecvf.com/content/CVPR2022/papers/Revaud_PUMP_Pyramidal_and_Uniqueness_Matching_Priors_for_Unsupervised_Learning_of_CVPR_2022_paper.pdf',
        'tldr': 'Unsupervised learning of local descriptors thanks to loss that incites unique matches.',
    },
    {
        'teaser': 'super_features.gif',
        'title': 'Learning Super-Features for Image Retrieval',
        'authors': 'Philippe Weinzaepfel, Thomas Lucas, Diane Larlus, Yannis Kalantidis',
        'where': 'ICLR 2022',
        'arxiv': 'https://arxiv.org/abs/2201.13182',
        'github': 'https://github.com/naver/fire',
        'tldr': 'Extract mid-level features for image retrieval with ASMK.',
        'star': True,
    },
    {
        'teaser': 'barelysupervised.jpg',
        'title': 'Barely-supervised learning: Semi-supervised learning with very few labeled images',
        'authors': 'Thomas Lucas, Philippe Weinzaepfel, Gregory Rogez',
        'where': 'AAAI 2022',
        'arxiv': 'https://arxiv.org/abs/2112.12004',
        'tldr': 'Use self-supervised learning if the pseudo-label from the weak augmentation is not confident enough.',
    },
    2021,
    {
        'teaser': 'posebert.jpg',
        'title': 'Leveraging MoCap Data for Human Mesh Recovery',
        'authors': 'Fabien Baradel, Thibault Groueix, Philippe Weinzaepfel, Romain Brégier, Yannis Kalantidis, Grégory Rogez',
        'where': '3DV 2021',
        'arxiv': 'https://arxiv.org/abs/2110.09243',
        'github': 'https://github.com/naver/posebert',
        'tldr': 'Motion capture data helps to improve image-based and video-based human mesh recovery.',
    },
    {
        'teaser': 'multifingan.jpg',
        'title': 'Multi-FinGAN: Generative Coarse-To-Fine Sampling of Multi-Finger Grasps',
        'authors': 'Jens Lundell, Enric Corona, Tran Nguyen Le, Francesco Verdoja, Philippe Weinzaepfel, Grégory Rogez, Francesc Moreno-Noguer, Ville Kyrki',
        'where': 'ICRA 2021',
        'arxiv': 'https://arxiv.org/abs/2012.09696',
        'github': 'https://github.com/aalto-intelligent-robotics/Multi-FinGAN',
        'tldr': 'A fast generative multi-finger grasp sampling method that synthesizes high quality grasps directly from RGB-D images in about a second.',
    },
    {
        'teaser': 'NLdataset.jpg',
        'title': 'Large-scale localization datasets in crowded indoor spaces',
        'authors': 'Donghwan Lee, Soohyun Ryu, Suyong Yeon, Yonghan Lee, Deokhwa Kim, Cheolho Han, Yohann Cabon, Philippe Weinzaepfel, Nicolas Guérin, Gabriela Csurka, Martin Humenberger',
        'where': 'CVPR 2021',
        'arxiv': 'https://arxiv.org/abs/2105.08941',
        'project': 'https://europe.naverlabs.com/blog/first-of-a-kind-large-scale-localization-datasets-in-crowded-indoor-spaces/',
        'dataset': 'https://www.naverlabs.com/en/datasets',
        'tldr': 'Dataset and baseline for large-scale localization in crowded indoor spaces (metro station or shopping mall).'
    },
    {
        'teaser': 'mimetics.jpg',
        'title': 'Mimetics: Towards Understanding Human Actions Out of Context',
        'authors': 'Philippe Weinzaepfel, Grégory Rogez',
        'where': 'IJCV 2021',
        'arxiv': 'https://arxiv.org/abs/1912.07249',
        'dataset': 'https://europe.naverlabs.com/research/computer-vision/mimetics/',
        'project': 'https://europe.naverlabs.com/blog/towards-understanding-human-actions-out-of-context-with-the-mimetics-dataset/',
        'tldr': 'The Mimetics dataset contains 713 video clips of mimed action to evaluate out-of-context human action methods.',
    },
    2020,
    {
        'teaser': 'mochi.png',
        'title': 'Hard negative mixing for contrastive learning',
        'authors': 'Yannis Kalantidis, Mert Bulent Sariyildiz, Noe Pion, Philippe Weinzaepfel, Diane Larlus',
        'where': 'NeurIPS 2020',
        'arxiv': 'https://arxiv.org/abs/2010.01028',
        'project': 'https://europe.naverlabs.com/research/publications/hard-negative-mixing-for-contrastive-learning/',
        'tldr': 'Generating hard negatives in the feature space for improved self-supervised contrastive learning.'
    },
    {
        'teaser': 'superloss.png',
        'title': 'SuperLoss: A Generic Loss for Robust Curriculum Learning',
        'authors': 'Thibault Castells, Philippe Weinzaepfel, Jerome Revaud',
        'where': 'NeurIPS 2020',
        'paper': 'https://proceedings.neurips.cc/paper/2020/file/2cfa8f9e50e0f510ede9d12338a5f564-Paper.pdf',
        'tldr': 'Automatically downweighting samples with a high loss implicitly performs curriculum learning.',
    },
    {
        'teaser': 'mannequin.jpg',
        'title': 'SMPLy Benchmarking 3D Human Pose Estimation in the Wild',
        'authors': 'Vincent Leroy, Philippe Weinzaepfel, Romain Brégier, Hadrien Combaluzier, Grégory Rogez',
        'where': '3DV 2020',
        'arxiv': 'https://arxiv.org/abs/2012.02743',
        'dataset': 'https://europe.naverlabs.com/research/publications/smply-benchmarking-3d-human-pose-estimation-in-the-wild/',
        'tldr': 'Dataset for in-the-wild human mesh recovery evaluation by fitting pseudo-ground-truth on Mannequin Challenge videos where people are static.',
    },
    {
        'teaser': 'dope.gif',
        'title': 'DOPE: Distillation Of Part Experts for whole-body 3D pose estimation in the wild',
        'authors': 'Philippe Weinzaepfel, Romain Brégier, Hadrien Combaluzier, Vincent Leroy, Grégory Rogez',
        'where': 'ECCV 2020',
        'arxiv': 'https://arxiv.org/abs/2008.09457',
        'github': 'https://github.com/naver/dope',
        'tldr': 'A novel, efficient model for whole-body 3D pose estimation (including bodies, hands and faces),  trained by mimicking the output of hand-, body- and face-pose experts.',
        'star': True,
    },
    {
        'teaser': 'hands2019.jpg',
        'title': 'Measuring Generalisation to Unseen Viewpoints, Articulations, Shapes and Objects for 3D Hand Pose Estimation under Hand-Object Interaction',
        'authors': 'Anil Armagan, Guillermo Garcia-Hernando, Seungryul Baek, Shreyas Hampali, Mahdi Rad, Zhaohui Zhang, Shipeng Xie, MingXiu Chen, Boshen Zhang, Fu Xiong, Yang Xiao, Zhiguo Cao, Junsong Yuan, Pengfei Ren, Weiting Huang, Haifeng Sun, Marek Hroz, Jakub Kanis, Zdenek Krnoul, Qingfu Wan, Shile Li, Linlin Yang, Dongheui Lee, Angela Yao, Weiguo Zhou, Sijia Mei, Yunhui Liu, Adrian Spurr, Umar Iqbal, Pavlo Molchanov, Philippe Weinzaepfel, Romain Brégier, Grégory Rogez, Vincent Lepetit, Tae-Kyun Kim',
        'where': 'ECCV 2020',
        'arxiv': 'https://arxiv.org/abs/2003.13764',
        'project': 'https://sites.google.com/view/hands2019/challenge',
        'tldr': "Outcome of the HANDS'19 challenge.",
    },
    2019,
    {
        'teaser': 'r2d2.webp',
        'title': 'R2D2: Repeatable and Reliable Detector and Descriptor',
        'authors': 'Jerome Revaud, Cesar De Souza, Martin Humenberger, Philippe Weinzaepfel',
        'where': 'NeurIPS 2019 (oral)',
        'arxiv': 'https://arxiv.org/abs/1906.06195',
        'github': 'https://github.com/naver/r2d2',
        'tldr': 'A neural network trains to detect and describe repeatable and reliable keypoints',        
        'star': True,
    },
    {
        'teaser': 'mars.png',
        'title': 'MARS: Motion-Augmented RGB Stream for Action Recognition',
        'authors': 'Nieves Crasto, Philippe Weinzaepfel, Karteek Alahari, Cordelia Schmid',
        'where': 'CVPR 2019',
        'paper': 'https://openaccess.thecvf.com/content_CVPR_2019/papers/Crasto_MARS_Motion-Augmented_RGB_Stream_for_Action_Recognition_CVPR_2019_paper.pdf',
        'github': 'https://github.com/craston/MARS',
        'tldr': 'Distill an optical flow based action recognition network into a RGB-based network.',
    },
    {
        'teaser': 'oois.webp',
        'title': 'Visual Localization by Learning O-of-Interest Dense Match Regression',
        'authors': 'Philippe Weinzaepfel, Gabriela Csurka, Yohann Cabon, Martin Humenberger',
        'where': 'CVPR 2019',
        'paper': 'https://openaccess.thecvf.com/content_CVPR_2019/papers/Weinzaepfel_Visual_Localization_by_Learning_Objects-Of-Interest_Dense_Match_Regression_CVPR_2019_paper.pdf',
        'project': 'https://europe.naverlabs.com/blog/visual-localization-by-learning-oois-dense-match-regression/',
        'dataset': 'https://europe.naverlabs.com/research/3d-vision/virtual-gallery-dataset/',
        'tldr': 'Visual localization by predicting dense texture coordinates of a given list of planar objects.' 
    },
    {
        'teaser': 'LCRNetpp.jpg',
        'title': 'LCR-Net++: Multi-person 2D and 3D Pose Detection in Natural Images',
        'authors': 'Gregory Rogez, Philippe Weinzaepfel, Cordelia Schmid',
        'where': 'IEEE Trans. PAMI 2019',
        'arxiv': 'https://arxiv.org/abs/1803.00455',
        'github': 'https://github.com/naver/lcrnet-v2-improved-ppi',
        'tldr': 'Journal extension of LCR-Net (CVPR 2017) for robust 2D-3D human pose estimation in the wild.',
        'star': True,
        'confdemo': 'CVPR 2018',
    },
    2018,
    {
        'teaser': 'potion.jpg',
        'title': 'PoTion: Pose MoTion Representation for Action Recognition',
        'authors': 'Vasileios Choutas, Philippe Weinzaepfel, Jérôme Revaud, Cordelia Schmid',
        'where': 'CVPR 2018',
        'paper': 'https://openaccess.thecvf.com/content_cvpr_2018/papers/Choutas_PoTion_Pose_MoTion_CVPR_2018_paper.pdf',
        'tldr': 'Action recognition from human poses using body-joint heatmaps that are colored according to their motions.',
    },    
    2017,
    {
        'teaser': 'act.jpg',
        'title': 'Action Tubelet Detector for Spatio-Temporal Action Localization',
        'authors': 'Vicky Kalogeiton, Philippe Weinzaepfel, Vittorio Ferrari, Cordelia Schmid',
        'where': 'ICCV 2017',
        'arxiv': 'https://arxiv.org/abs/1705.01861',
        'tldr': 'Spatio-temporal video action detector by regressing tubelets from anchor cuboids.',
    },
    {
        'teaser': 'objact.jpg',
        'title': 'Joint Learning of Object and Action Detectors',
        'authors': 'Vicky Kalogeiton, Philippe Weinzaepfel, Vittorio Ferrari, Cordelia Schmid',
        'where': 'ICCV 2017',
        'paper': 'https://openaccess.thecvf.com/content_ICCV_2017/papers/Kalogeiton_Joint_Learning_of_ICCV_2017_paper.pdf',
        'tldr': 'Spatio-temporal detection of different action classes performed by various types of "objects" in videos.'
    },
    {
        'teaser': 'LCRNet.jpg',
        'title': 'LCR-Net: Localization-Classification-Regression for Human Pose',
        'authors': 'Gregory Rogez, Philippe Weinzaepfel, Cordelia Schmid',
        'where': 'CVPR 2017 (spotlight)',
        'paper': 'https://openaccess.thecvf.com/content_cvpr_2017/papers/Rogez_LCR-Net_Localization-Classification-Regression_for_CVPR_2017_paper.pdf',
        'tldr': 'Highly-robust 2D-3D human pose estimation by classifying poses among some predefined clusters and regressing the offset.',
    },
    2016,
    {
        'teaser': 'daly.jpg',
        'title': 'Human Action Localization with Sparse Spatial Supervision',
        'authors': 'Philippe Weinzaepfel, Xavier Martin, Cordelia Schmid',
        'where': 'arXiv 2016',
        'arxiv': 'https://arxiv.org/abs/1605.05197',
        'dataset': 'https://thoth.inrialpes.fr/daly/',
        'tldr': 'Spatio-temporal video action detection from temporal action annotation and one bounding box.',
    },
    {
        'teaser': 'deepmatching.jpg',
        'title': 'DeepMatching: Hierarchical Deformable Dense Matching',
        'authors': 'Jerome Revaud, Philippe Weinzaepfel, Zaid Harchaoui, Cordelia Schmid',
        'where': 'IJCV 2016',
        'arxiv': 'https://arxiv.org/abs/1506.07656',
        'code': 'https://thoth.inrialpes.fr/src/deepmatching.html',
        'project': 'https://thoth.inrialpes.fr/src/deepmatching.html',
        'tldr': 'A learning-free algorithm to compute dense correspondences with a hierarchical, multi-layer correlational architecture inspired by deep convolutional networks.',
    },
    {
        'teaser': 'phd.jpg',
        'title': 'Motion in action : optical flow estimation and action localization in videos',
        'authors': 'Philippe Weinzaepfel',
        'where': 'PhD Thesis, University Grenoble Alpes, 2016',
        'paper': 'https://theses.hal.science/tel-01407258/file/WEINZAEPFEL_2016_diffusion.pdf',
        'tldr': 'My PhD thesis.',
    },
    2015,
    {
        'teaser': 'learningtotrack.jpg',
        'title': 'Learning to Track for Spatio-Temporal Action Localization',
        'authors': 'Philippe Weinzaepfel, Zaid Harchaoui, Cordelia Schmid',
        'where': 'ICCV 2015',
        'arxiv': 'https://arxiv.org/abs/1506.01929',
        'tldr': 'Spatio-temporal video action detection by frame-level detection and scoring, tracking best candidates across videos and scoring the obtained tracks.',
    
    },
    {
        'teaser': 'mobo.jpg',
        'title': 'Learning to Detect Motion Boundaries',
        'authors': 'Philippe Weinzaepfel, Jerome Revaud, Zaid Harchaoui, Cordelia Schmid',
        'where': 'CVPR 2015',
        'paper': 'https://openaccess.thecvf.com/content_cvpr_2015/papers/Weinzaepfel_Learning_to_Detect_2015_CVPR_paper.pdf',
        'dataset': 'https://lear.inrialpes.fr/research/motionboundaries/',
        'code': 'https://lear.inrialpes.fr/research/motionboundaries/',
        'tldr': 'Dataset for motion boundary detection and structured random forest baseline.',
    },
    {
        'teaser': 'epicflow.jpg',
        'title': 'EpicFlow: Edge-Preserving Interpolation of Correspondences for Optical Flow',
        'authors': 'Jerome Revaud, Philippe Weinzaepfel, Zaid Harchaoui, Cordelia Schmid',
        'where': 'CVPR 2015 (oral)',
        'arxiv': 'https://arxiv.org/abs/1501.02565',
        'code': 'https://thoth.inrialpes.fr/src/epicflow/',
        'tldr': 'See title.',
        'star': True,
    },
    2013,
    {
        'teaser': 'deepflow.jpg',
        'title': 'DeepFlow: Large Displacement Optical Flow with Deep Matching',
        'authors': 'Philippe Weinzaepfel, Jerome Revaud, Zaid Harchaoui, Cordelia Schmid',
        'where': 'ICCV 2013 (oral)',
        'paper': 'https://openaccess.thecvf.com/content_iccv_2013/papers/Weinzaepfel_DeepFlow_Large_Displacement_2013_ICCV_paper.pdf',
        'code': 'https://thoth.inrialpes.fr/src/deepmatching.html',
        'tldr': 'Novel matching algorithm based on optimal hierarchical motion of moving quadrants tailored for large displacements and its application to optical flow.',
        'star': True,
    },
    2011,
    {
        'teaser': 'reconstructlocal.jpg',
        'title': 'Reconstructing an Image from its Local Descriptors',
        'authors': 'Philippe Weinzaepfel, Hervé Jégou, Patrick Pérez',
        'where': 'CVPR 2011',
        'paper': 'https://inria.hal.science/inria-00566718/PDF/weinzaepfel_cvpr11.pdf',
        'tldr': 'From the local features and their locations of an input image, the content can be reconstructed by looking from similar regions from a database.'
    },
]

def check_publication_list():
    for p in publication_list:
        if isinstance(p,int): continue
        assert 'teaser' in p, p
        assert os.path.isfile('assets/publications/'+p['teaser']), p
        assert 'title' in p, p
        assert 'authors' in p, p
        assert 'where' in p, p
        assert 'arxiv' in p or 'paper' in p, p
        assert 'tldr' in p
        invalid_keys = [k for k in p if k not in ['teaser','title','authors','where','arxiv','paper','tldr', 'code','github','dataset','demo','huggingface','winner','project','star','confdemo']]
        assert len(invalid_keys)==0, (invalid_keys,p)

def print_all_publis_html(nspaces=8):
    """print('<!DOCTYPE html>')
    print('<html lang="en">')
    print('<head>')
    print('  <link rel="stylesheet" href="style.css">')
    print('</head>')
    print('<body>')
    print('<div class="publications">')"""
    S = ' '*nspaces
    for p in publication_list:
        if p == 2024: break
        if isinstance(p, int):
            print(S+f'<h4>{p}</h4>')
            continue
        print(S+f'<div class="rounded-shadow-box onepubli{" onepubli-star" if "star" in p and p["star"] else ""}">')
        print(S+f'  <div class="onepubli-image">')
        print(S+f'    <img src="assets/publications/{p["teaser"]}"/>')
        print(S+f'  </div>')
        print(S+f'  <div class="onepubli-infos">')
        print(S+f'    <div class="onepubli-title">{p["title"]}</div>')
        print(S+f'    <div class="onepubli-author">{p["authors"].replace("é","&eacute").replace("á","&aacute").replace("ô","&ocirc;")}</div>')
        print(S+f'    <div class="onepubli-where">{p["where"]}</div>')
        print(S+f'    <div class="onepubli-links">')
        if 'arxiv' in p:
            print(S+f'      <span class="button-link"><a href="{p["arxiv"]}"><svg class="button-link-icon"><use href="#arxiv-icon"></use></svg><span>arXiv</span></a></span>')
        if 'paper' in p:
            print(S+f'      <span class="button-link"><a href="{p["paper"]}"><svg class="button-link-icon"><use href="#pdf-icon"></use></svg>paper</a></span>')
        if 'github' in p:
            print(S+f'      <span class="button-link"><a href="{p["github"]}"><svg class="button-link-icon"><use href="#github-icon"></use></svg><span>github</span></a></span>')
        if 'huggingface' in p:
            print(S+f'      <span class="button-link"><a href="{p["huggingface"]}"><svg class="button-link-icon"><use href="#huggingface-icon"></use></svg><span>demo</span></a></span>')
        if 'demo' in p:
            print(S+f'      <span class="button-link"><a href="{p["demo"]}"><svg class="button-link-icon"><use href="#demo-icon"></use></svg>demo</a></span>')
        if 'dataset' in p:
            print(S+f'      <span class="button-link"><a href="{p["dataset"]}"><svg class="button-link-icon"><use href="#database-icon"></use></svg>dataset</a></span>')
        if 'project' in p:
            print(S+f'      <span class="button-link"><a href="{p["project"]}"><svg class="button-link-icon"><use href="#website-icon"></use></svg>project</a></span>')
        if 'winner' in p:
            print(S+f'      <span class="button-link"><a href="{p["winner"][0]}"><svg class="button-link-icon"><use href="#winner-icon"></use></svg>{p["winner"][1]}</a>')
        if 'star' in p: pass
        if 'confdemo' in p: pass
        print(S+f'    </div>')
        print(S+f'    <div class="onepubli-tldr">{p["tldr"]}</div>')
        print(S+f'  </div>')
        print(S+f'</div>')
    """print('  </div>')
    print('</body>')
    print('</html>')"""
        

def print_tldr():
    for p in publication_list:
        if isinstance(p, int):
            continue
        print(p['tldr'])

if __name__=="__main__":
    check_publication_list()
    print_all_publis_html()