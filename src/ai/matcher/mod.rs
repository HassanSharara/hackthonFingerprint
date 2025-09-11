
use opencv::{
    core::{KeyPoint, Vector, Mat, no_array, Size, DMatch, NORM_L2},
    features2d::{BFMatcher, FlannBasedMatcher, SIFT},
    prelude::*,
    imgcodecs,
    imgproc,
};
use std::fs;
use std::path::Path;

/// BruteForce matcher version with CORRECT parameter order
pub fn match_using_sift_bruteforce(image_path: &str, folder_to_lookat: &str) -> opencv::Result<(Option<String>, i32)> {
    // Create SIFT detector
    let mut sift = SIFT::create(0, 3, 0.04, 10.0, 1.6)?;

    // Load query image
    let query_img = imgcodecs::imread(image_path, imgcodecs::IMREAD_GRAYSCALE)?;
    if query_img.empty() {
        return Err(opencv::Error::new(
            opencv::core::StsError,
            format!("Image not found: {}", image_path)
        ));
    }

    // Resize query image to 128x128
    let mut resized_query = Mat::default();
    imgproc::resize(&query_img, &mut resized_query, Size::new(128, 128), 0.0, 0.0, imgproc::INTER_LINEAR)?;

    // Extract keypoints and descriptors from query image
    let mut kp1 = Vector::<KeyPoint>::new();
    let mut des1 = Mat::default();
    sift.detect_and_compute(&resized_query, &no_array(), &mut kp1, &mut des1, false)?;

    if des1.empty() {
        return Err(opencv::Error::new(
            opencv::core::StsError,
            "No descriptors found in query image".to_string()
        ));
    }

    let mut best_match: Option<String> = None;
    let mut best_score = -1;

    // Process database folder
    let entries = fs::read_dir(folder_to_lookat)
        .map_err(|e| opencv::Error::new(
            opencv::core::StsError,
            format!("Failed to read directory {}: {}", folder_to_lookat, e)
        ))?;

    for entry in entries {
        let path = entry
            .map_err(|e| opencv::Error::new(
                opencv::core::StsError,
                format!("Failed to read directory entry: {}", e)
            ))?
            .path();

        if !path.is_file() { continue; }

        let path_str = match path.to_str() {
            Some(s) => s,
            None => continue,
        };

        // Skip non-image files
        if !path_str.to_lowercase().ends_with(".bmp") &&
            !path_str.to_lowercase().ends_with(".jpg") &&
            !path_str.to_lowercase().ends_with(".png") {
            continue;
        }

        // Load database image
        let db_img = match imgcodecs::imread(path_str, imgcodecs::IMREAD_GRAYSCALE) {
            Ok(img) if !img.empty() => img,
            _ => continue,
        };

        // Resize database image to 128x128
        let mut resized_db = Mat::default();
        if imgproc::resize(&db_img, &mut resized_db, Size::new(128, 128), 0.0, 0.0, imgproc::INTER_LINEAR).is_err() {
            continue;
        }

        // Extract keypoints and descriptors from database image
        let mut kp2 = Vector::<KeyPoint>::new();
        let mut des2 = Mat::default();
        if sift.detect_and_compute(&resized_db, &no_array(), &mut kp2, &mut des2, false).is_err() || des2.empty() {
            continue;
        }

        // Create BruteForce matcher for this image
        let mut bf = BFMatcher::create(NORM_L2, false)?;

        // Perform direct matching using BruteForce
        let mut matches = Vector::<Vector<DMatch>>::new();

        // Correct knn_match signature: (query_descriptors, matches, k, masks, compact_result)
        if bf.knn_match(
            &des1,          // query descriptors
            &mut matches,   // matches output
            2,              // k
            &des2,          // train descriptors (used as mask parameter)
            false           // compact_result
        ).is_err() {
            continue;
        }

        // Apply Lowe's ratio test
        let mut good_matches = 0;

        for match_pair in matches.iter() {
            if match_pair.len() == 2 {
                if let (Ok(m), Ok(n)) = (match_pair.get(0), match_pair.get(1)) {
                    println!("{} {}",m.distance,n.distance);
                    if m.distance < 0.7 * n.distance {
                        good_matches += 1;
                    }
                }
            }
        }

        // Update best match if this score is better
        if good_matches > best_score {
            best_score = good_matches;
            best_match = Some(path.file_name().unwrap().to_string_lossy().to_string());
        }
    }

    // Print result
    let query_basename = Path::new(image_path)
        .file_name()
        .unwrap_or_default()
        .to_string_lossy();

    // println!("🔍 Best match for {}: {:?} (score={})", query_basename, best_match, best_score);

    Ok((best_match, best_score))
}

/// Alternative approach using match() instead of knn_match for simpler usage
pub fn match_using_sift_simple(image_path: &str, folder_to_lookat: &str) -> opencv::Result<(Option<String>, i32)> {
    // Create SIFT detector
    let mut sift = SIFT::create(0, 3, 0.04, 10.0, 1.6)?;

    // Load query image
    let query_img = imgcodecs::imread(image_path, imgcodecs::IMREAD_GRAYSCALE)?;
    if query_img.empty() {
        return Err(opencv::Error::new(
            opencv::core::StsError,
            format!("Image not found: {}", image_path)
        ));
    }

    // Resize query image to 128x128
    let mut resized_query = Mat::default();
    imgproc::resize(&query_img, &mut resized_query, Size::new(128, 128), 0.0, 0.0, imgproc::INTER_LINEAR)?;

    // Extract keypoints and descriptors from query image
    let mut kp1 = Vector::<KeyPoint>::new();
    let mut des1 = Mat::default();
    sift.detect_and_compute(&resized_query, &no_array(), &mut kp1, &mut des1, false)?;

    if des1.empty() {
        return Err(opencv::Error::new(
            opencv::core::StsError,
            "No descriptors found in query image".to_string()
        ));
    }

    // Create BruteForce matcher
    let mut bf = BFMatcher::create(NORM_L2, false)?;

    let mut best_match: Option<String> = None;
    let mut best_score = -1;

    // Process database folder
    let entries = fs::read_dir(folder_to_lookat)
        .map_err(|e| opencv::Error::new(
            opencv::core::StsError,
            format!("Failed to read directory {}: {}", folder_to_lookat, e)
        ))?;

    for entry in entries {
        let path = entry
            .map_err(|e| opencv::Error::new(
                opencv::core::StsError,
                format!("Failed to read directory entry: {}", e)
            ))?
            .path();

        if !path.is_file() { continue; }

        let path_str = match path.to_str() {
            Some(s) => s,
            None => continue,
        };

        // Skip non-image files
        if !path_str.to_lowercase().ends_with(".bmp") &&
            !path_str.to_lowercase().ends_with(".jpg") &&
            !path_str.to_lowercase().ends_with(".png") {
            continue;
        }

        // Load database image
        let db_img = match imgcodecs::imread(path_str, imgcodecs::IMREAD_GRAYSCALE) {
            Ok(img) if !img.empty() => img,
            _ => continue,
        };

        // Resize database image to 128x128
        let mut resized_db = Mat::default();
        if imgproc::resize(&db_img, &mut resized_db, Size::new(128, 128), 0.0, 0.0, imgproc::INTER_LINEAR).is_err() {
            continue;
        }

        // Extract keypoints and descriptors from database image
        let mut kp2 = Vector::<KeyPoint>::new();
        let mut des2 = Mat::default();
        if sift.detect_and_compute(&resized_db, &no_array(), &mut kp2, &mut des2, false).is_err() || des2.empty() {
            continue;
        }

        // Use simple match() method instead of knn_match
        let mut matches = Vector::<DMatch>::new();
        if bf.train_match(&des1, &des2, &mut matches, &no_array()).is_err() {
            continue;
        }

        // Count good matches with distance threshold
        let mut good_matches = 0;
        let distance_threshold = 120.0; // Adjust this threshold as needed

        for i in 0..matches.len() {
            if let Ok(m) = matches.get(i) {
                if m.distance < distance_threshold {
                    good_matches += 1;
                }
            }
        }

        // Update best match if this score is better
        if good_matches > best_score {
            best_score = good_matches;
            best_match = Some(path.file_name().unwrap().to_string_lossy().to_string());
        }
    }

    // Print result
    let query_basename = Path::new(image_path)
        .file_name()
        .unwrap_or_default()
        .to_string_lossy();

    // println!("🔍 Best match for {}: {:?} (score={})", query_basename, best_match, best_score);

    Ok((best_match, best_score))
}

/// FLANN matcher version with knn_match + Lowe's ratio test
pub fn match_using_sift_flann(image_path: &str, folder_to_lookat: &str) -> opencv::Result<(Option<String>, i32)> {
    // Create SIFT detector
    let mut sift = SIFT::create(0, 3, 0.04, 10.0, 1.6)?;

    // Load query image
    let query_img = imgcodecs::imread(image_path, imgcodecs::IMREAD_GRAYSCALE)?;
    if query_img.empty() {
        return Err(opencv::Error::new(
            opencv::core::StsError,
            format!("Image not found: {}", image_path)
        ));
    }

    // Resize query image to 128x128
    let mut resized_query = Mat::default();
    imgproc::resize(&query_img, &mut resized_query, Size::new(128, 128), 0.0, 0.0, imgproc::INTER_LINEAR)?;

    // Extract keypoints and descriptors from query image
    let mut kp1 = Vector::<KeyPoint>::new();
    let mut des1 = Mat::default();
    sift.detect_and_compute(&resized_query, &no_array(), &mut kp1, &mut des1, false)?;

    if des1.empty() {
        return Err(opencv::Error::new(
            opencv::core::StsError,
            "No descriptors found in query image".to_string()
        ));
    }

    let mut best_match: Option<String> = None;
    let mut best_score = -1;

    // Process database folder
    let entries = fs::read_dir(folder_to_lookat)
        .map_err(|e| opencv::Error::new(
            opencv::core::StsError,
            format!("Failed to read directory {}: {}", folder_to_lookat, e)
        ))?;

    for entry in entries {
        let path = entry
            .map_err(|e| opencv::Error::new(
                opencv::core::StsError,
                format!("Failed to read directory entry: {}", e)
            ))?
            .path();

        if !path.is_file() { continue; }

        let path_str = match path.to_str() {
            Some(s) => s,
            None => continue,
        };

        // Skip non-image files
        if !path_str.to_lowercase().ends_with(".bmp") &&
            !path_str.to_lowercase().ends_with(".jpg") &&
            !path_str.to_lowercase().ends_with(".png") {
            continue;
        }

        // Load database image
        let db_img = match imgcodecs::imread(path_str, imgcodecs::IMREAD_GRAYSCALE) {
            Ok(img) if !img.empty() => img,
            _ => continue,
        };

        // Resize database image to 128x128
        let mut resized_db = Mat::default();
        if imgproc::resize(&db_img, &mut resized_db, Size::new(128, 128), 0.0, 0.0, imgproc::INTER_LINEAR).is_err() {
            continue;
        }

        // Extract keypoints and descriptors from database image
        let mut kp2 = Vector::<KeyPoint>::new();
        let mut des2 = Mat::default();
        if sift.detect_and_compute(&resized_db, &no_array(), &mut kp2, &mut des2, false).is_err() || des2.empty() {
            continue;
        }

        // Create FLANN matcher
        let mut flann = FlannBasedMatcher::create()?;
        let mut knn_matches = Vector::<Vector<DMatch>>::new();

        // Run knn_match (k=2)
        if flann.knn_train_match(&des1, &des2, &mut knn_matches, 2, &no_array(), false).is_err() {
            continue;
        }

        // Lowe’s ratio test
        let mut good_matches = 0;
        let ratio_thresh = 0.7;

        for i in 0..knn_matches.len() {
            if let Ok(m) = knn_matches.get(i) {
                if m.len() >= 2 {
                    let m1 = m.get(0)?;
                    let m2 = m.get(1)?;
                    if m1.distance < ratio_thresh * m2.distance {
                        good_matches += 1;
                    }
                }
            }
        }

        // Update best match if this score is better
        if good_matches > best_score {
            best_score = good_matches;
            best_match = Some(path.file_name().unwrap().to_string_lossy().to_string());
        }
    }

    // Print result
    let query_basename = Path::new(image_path)
        .file_name()
        .unwrap_or_default()
        .to_string_lossy();

    // println!("🔍 Best match for {}: {:?} (score={})", query_basename, best_match, best_score);

    Ok((best_match, best_score))
}

/// Main function - uses simple BruteForce by default
pub fn match_using_sift(image_path: &str, folder_to_lookat: &str) -> opencv::Result<(Option<String>, i32)> {
    match_using_sift_flann(image_path, folder_to_lookat)
}
