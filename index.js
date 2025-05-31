// index.js
const { SecretManagerServiceClient } = require('@google-cloud/secret-manager');
const { createClient }              = require('@supabase/supabase-js');
const { Storage }                   = require('@google-cloud/storage');

const PROJECT_ID = 'fake-profile-detection-460117';
const secretClient = new SecretManagerServiceClient();

/**
 * Fetch the latest Supabase service_role key from Secret Manager
 */
async function getSupabaseKey() {
  const name = `projects/${PROJECT_ID}/secrets/supabase-service-role-key/versions/latest`;
  const [version] = await secretClient.accessSecretVersion({ name });
  return version.payload.data.toString('utf8');
}

/**
 * HTTP Cloud Function triggered by Supabase webhook
 */
exports.syncToGCS = async (req, res) => {
  try {
    // 1) Grab the secret
    const serviceKey = await getSupabaseKey();

    // 2) Init Supabase client with service_role key
    const sb = createClient(
      process.env.SUPABASE_URL,
      serviceKey
    );

    // 3) Read the webhook payload
    const { record } = req.body;
    const bucketId = record.bucket_id;
    const fileName = record.name;

    // 4) Download the new file from Supabase Storage
    const { data, error } = await sb
      .storage
      .from(bucketId)
      .download(fileName);

    if (error) throw error;

    // 5) Upload it to your GCS bucket
    const gcs    = new Storage();
    const dest   = process.env.GCS_BUCKET;
    const file   = gcs.bucket(dest).file(fileName);
    
    // Convert Blob to Buffer
    const arrayBuffer = await data.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    await file.save(buffer);

    // // 5) Upload it to your GCS bucket
    // const gcs    = new Storage();
    // const dest   = process.env.GCS_BUCKET;
    // const file   = gcs.bucket(dest).file(fileName);
    // await file.save(await data.arrayBuffer());

    res.status(200).send('OK');
  } catch (err) {
    console.error(err);
    res.status(500).send('Error: ' + err.message);
  }
};

